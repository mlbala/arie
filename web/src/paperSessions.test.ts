import { describe, expect, it } from "vitest";
import {
  EMPTY_SESSIONS,
  loadSessions,
  MAX_HISTORY,
  MAX_PINNED,
  paperSessionsReducer,
  saveSessions,
  type PaperSessionAction,
  type PaperSessions,
} from "./paperSessions";

const run = (actions: PaperSessionAction[], start: PaperSessions = EMPTY_SESSIONS) =>
  actions.reduce(paperSessionsReducer, start);
const open = (...ids: string[]) => ids.map((id): PaperSessionAction => ({ type: "open", id }));
const ids = (count: number, prefix = "p") => Array.from({ length: count }, (_, index) => `${prefix}${index}`);

describe("paperSessionsReducer", () => {
  it("pins up to five papers in opening order without reordering on revisit", () => {
    const state = run([...open("a", "b", "c"), ...open("a")]);
    expect(state.pinned).toEqual(["a", "b", "c"]);
    expect(state.activeId).toBe("a");
    expect(state.history).toEqual([]);
  });

  it("shifts the oldest pinned paper into history when a sixth is opened", () => {
    const state = run(open(...ids(MAX_PINNED + 1)));
    expect(state.pinned).toEqual(["p1", "p2", "p3", "p4", "p5"]);
    expect(state.history).toEqual(["p0"]);
    expect(state.activeId).toBe("p5");
  });

  it("caps history at ten, discarding the oldest shifted papers", () => {
    const state = run(open(...ids(MAX_PINNED + MAX_HISTORY + 3)));
    expect(state.pinned).toHaveLength(MAX_PINNED);
    expect(state.history).toHaveLength(MAX_HISTORY);
    expect(state.history[0]).toBe("p12");
    expect(state.history.at(-1)).toBe("p3");
  });

  it("restores a history paper to the pinned pane and shifts the oldest out", () => {
    const state = run([...open(...ids(6)), ...open("p0")]);
    expect(state.pinned).toEqual(["p2", "p3", "p4", "p5", "p0"]);
    expect(state.history).toEqual(["p1"]);
  });

  it("closes a tab into history and activates its neighbour", () => {
    const state = run([...open("a", "b", "c"), ...open("b"), { type: "close", id: "b" }]);
    expect(state.pinned).toEqual(["a", "c"]);
    expect(state.activeId).toBe("c");
    expect(state.history).toEqual(["b"]);
    expect(run([{ type: "close", id: "c" }, { type: "close", id: "a" }], state).activeId).toBeNull();
  });

  it("marks the session touched when the unpinned default paper is closed", () => {
    const state = paperSessionsReducer(EMPTY_SESSIONS, { type: "close", id: "default" });
    expect(state.touched).toBe(true);
    expect(state.history).toEqual(["default"]);
  });

  it("clears only history, and forgets deleted papers everywhere", () => {
    const shifted = run(open(...ids(7)));
    const cleared = paperSessionsReducer(shifted, { type: "clearHistory" });
    expect(cleared.history).toEqual([]);
    expect(cleared.pinned).toEqual(shifted.pinned);
    const forgotten = run([{ type: "forget", id: "p6" }, { type: "forget", id: "p0" }], shifted);
    expect(forgotten.pinned).not.toContain("p6");
    expect(forgotten.history).toEqual(["p1"]);
    expect(forgotten.activeId).toBeNull();
  });
});

it("restores saved tabs within limits and ignores unreadable storage", () => {
  saveSessions({ pinned: ["a", "b"], history: ["c"], activeId: "b", touched: true });
  expect(loadSessions()).toEqual({ pinned: ["a", "b"], history: ["c"], activeId: "b", touched: true });
  localStorage.setItem("arie.ui.sessions", JSON.stringify({ pinned: Array.from({ length: 9 }, (_, i) => `p${i}`), history: [1, "h"], activeId: 3 }));
  expect(loadSessions()).toEqual({ pinned: ["p0", "p1", "p2", "p3", "p4"], history: ["h"], activeId: null, touched: false });
  localStorage.setItem("arie.ui.sessions", "{not json");
  expect(loadSessions()).toEqual(EMPTY_SESSIONS);
});
