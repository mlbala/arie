import type { PaperGuide } from "./api/client";

/** Inline code that survives backticks in the equation itself. */
function code(text: string): string {
  const fence = text.includes("`") ? "``" : "`";
  return `${fence} ${text} ${fence}`;
}

function list(title: string, items: string[], ordered = false): string {
  if (!items.length) return "";
  const lines = items.map((item, index) => `${ordered ? `${index + 1}.` : "-"} ${item}`);
  return `### ${title}\n\n${lines.join("\n")}`;
}

/** The guide as Markdown, so its markers become the same citation chips as an explanation. */
export function guideMarkdown(guide: PaperGuide): string {
  const parts = [
    guide.tldr && `**TL;DR.** ${guide.tldr}`,
    guide.problem && `**Problem.** ${guide.problem}`,
    list("How it works", guide.method_steps, true),
    guide.key_equations.length ? `### Key equations\n\n${guide.key_equations.map((item) => [
      `- ${code(item.equation)}: ${item.meaning}`,
      ...item.symbols.map((symbol) => `  - ${code(symbol.term)}: ${symbol.meaning}`),
    ].join("\n")).join("\n")}` : "",
    list("Results", guide.results),
    list("Limitations", guide.limitations),
    list("Glossary", guide.glossary.map((item) => `**${item.term}**: ${item.meaning}`)),
    list("Sections", guide.section_summaries.map((item) => `**${item.section}**: ${item.summary}`)),
  ];
  return parts.filter(Boolean).join("\n\n");
}
