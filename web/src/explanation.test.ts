import { describe, expect, it } from "vitest";
import {
  mergeSentences,
  normalizeCitations,
  normalizeSentences,
  reasonLabel,
  renderExplanation,
  type SentenceStatus,
} from "./explanation";

const page = (html: string) => {
  const root = document.createElement("div");
  root.innerHTML = html;
  return root;
};
const sentence = (text: string, whole: string, status: string, reasons: string[] = []): SentenceStatus => {
  const start = whole.indexOf(text);
  return { start, end: start + text.length, text, status, reasons };
};

describe("renderExplanation", () => {
  it("escapes raw HTML and drops links and images the model writes", () => {
    const text = 'Use <img src=x onerror="alert(1)"> and <script>alert(2)</script> '
      + "with [a link](https://evil.example/?q=paper) and ![pixel](https://evil.example/p.png).";
    const root = page(renderExplanation(text, [], []));
    expect(root.querySelector("img, script, a")).toBeNull();
    expect(root.textContent).toContain('<img src=x onerror="alert(1)">');
    expect(root.textContent).toContain("<script>alert(2)</script>");
  });

  it("renders Markdown and turns markers into page chips once citations arrive", () => {
    const text = "**Method.** Workers read the paper [c3][c9].\n\n- Results improve [c4].";
    const citations = normalizeCitations([
      { marker: 3, chunk_id: "p:3", page: 2, char_start: 10, char_end: 90, checked: true },
      { marker: 4, chunk_id: "p:4", page: 5, char_start: 0, char_end: 40, checked: false },
    ]);
    const streaming = page(renderExplanation(text, [], undefined));
    expect(streaming.querySelector("strong")?.textContent).toBe("Method.");
    expect(Array.from(streaming.querySelectorAll("button")).map((chip) => chip.textContent)).toEqual(["c3", "c9", "c4"]);
    expect(streaming.querySelector("button")).toBeDisabled();

    const done = page(renderExplanation(text, [], citations));
    const chips = Array.from(done.querySelectorAll("button"));
    expect(chips.map((chip) => chip.textContent)).toEqual(["p. 2", "c9", "p. 5"]);
    expect(chips[0]).toHaveAttribute("data-marker", "3");
    expect(chips[0]).toHaveAttribute("aria-label", "Source c3, page 2");
    expect(chips[1]).toBeDisabled();
    expect(chips[1]).toHaveAttribute("aria-label", "Source c9, not a passage of this paper");
    expect(chips[2]).toHaveClass("unchecked");
    expect(done.querySelector("li")?.textContent).toBe("Results improve p. 5.");
  });

  it("styles each sentence by status, with reasons as its tooltip, keeping emphasis intact", () => {
    const text = "**The gain is 9.1 points [c1].** It is large [c2]. Transformers are common.";
    const sentences = [
      sentence("**The gain is 9.1 points [c1].**", text, "flagged", ["number_not_in_source", "low_grounding"]),
      sentence("It is large [c2].", text, "cited_ok"),
      sentence("Transformers are common.", text, "background"),
    ];
    const root = page(renderExplanation(text, sentences, []));
    const flagged = root.querySelector(".status-flagged");
    expect(flagged?.querySelector("strong")?.textContent).toContain("The gain is 9.1 points");
    expect(flagged).toHaveAttribute("title", "A number is not in the cited passage; low grounding");
    expect(root.querySelector(".status-cited_ok")).toHaveTextContent("It is large");
    expect(root.querySelector(".status-cited_ok")).not.toHaveAttribute("title");
    expect(root.querySelector(".status-background")).toHaveAttribute("title", "Background, not cited from the paper");
    expect(root.textContent).not.toMatch(/[⸜⸝]/);
  });

  it("wraps table cells and quoted lines without breaking their Markdown", () => {
    const text = "| Model | Score |\n| --- | --- |\n| Ours | 92.1 [c3] |\n\n> Quoted line [c5].";
    const row = sentence("| Ours | 92.1 [c3] |", text, "flagged", ["number_not_in_source"]);
    const quote = sentence("> Quoted line [c5].", text, "background");
    const root = page(renderExplanation(text, [row, quote], []));
    const cells = Array.from(root.querySelectorAll("tbody td"));
    expect(cells.map((cell) => cell.textContent)).toEqual(["Ours", "92.1 c3"]);
    expect(cells.every((cell) => cell.querySelector(".status-flagged"))).toBe(true);
    expect(root.querySelector("blockquote .status-background")).toHaveTextContent("Quoted line");
  });

  it("finds a sentence whose offsets drifted and skips one that is not in the text", () => {
    const text = "First claim [c1]. Second claim [c2].";
    const moved = { ...sentence("Second claim [c2].", text, "flagged", ["invalid_marker"]), start: 0, end: 18 };
    const absent = { start: 0, end: 5, text: "Nowhere.", status: "flagged", reasons: [] };
    const root = page(renderExplanation(text, [moved, absent], undefined));
    expect(root.querySelectorAll(".answer-sentence")).toHaveLength(1);
    expect(root.querySelector(".status-flagged")).toHaveTextContent("Second claim");
  });
});

it("parses sentence verdicts and keeps a flag when a later checker merges in", () => {
  const checks = normalizeSentences([
    { start: 0, end: 10, text: "First one.", status: "cited_ok", markers: [1], reasons: [] },
    { start: 11, end: 20, text: "Second.", status: "flagged", reasons: ["lost_qualifier", 3] },
    { start: "x", end: 5, text: "Bad.", status: "flagged" },
  ]);
  expect(checks).toHaveLength(2);
  expect(checks[1]!.reasons).toEqual(["lost_qualifier"]);
  const grounding = normalizeSentences([
    { start: 0, end: 10, text: "First one.", status: "flagged", reasons: ["low_grounding"] },
    { start: 11, end: 20, text: "Second.", status: "cited_ok", reasons: [] },
  ]);
  expect(mergeSentences(checks, grounding).map(({ status, reasons }) => [status, reasons])).toEqual([
    ["flagged", ["low_grounding"]],
    ["flagged", ["lost_qualifier"]],
  ]);
  expect(reasonLabel("lost_qualifier")).toBe("Drops a qualifier the cited passage has");
});
