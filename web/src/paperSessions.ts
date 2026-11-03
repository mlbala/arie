/** Active paper chats pinned in the top pane, and the sidebar history they shift into. */
export const MAX_PINNED = 5;
export const MAX_HISTORY = 10;

export interface PaperSessions {
  /** Pinned paper ids, oldest first; never longer than MAX_PINNED. */
  pinned: string[];
  /** Shifted or closed paper ids, most recent first; never longer than MAX_HISTORY. */
  history: string[];
  /** Explicitly chosen paper, or null when none is chosen. */
  activeId: string | null;
  /** True once the reader has opened a paper, which disables the default-paper fallback. */
  touched: boolean;
}

export type PaperSessionAction =
  | { type: "open"; id: string }
  | { type: "close"; id: string }
  | { type: "deselect" }
  | { type: "forget"; id: string }
  | { type: "clearHistory" };

export const EMPTY_SESSIONS: PaperSessions = { pinned: [], history: [], activeId: null, touched: false };

function remember(history: string[], ids: string[]) {
  return [...ids, ...history.filter((item) => !ids.includes(item))].slice(0, MAX_HISTORY);
}

/**
 * Pure transition function for paper sessions.
 *
 * - `open` pins a paper (restoring it from history if present); pinning a sixth paper
 *   shifts the oldest pinned paper into history (FIFO) instead of discarding it.
 * - `close` unpins a paper into history and activates its nearest neighbour tab; closing
 *   the unpinned default paper also marks the session touched so it does not reappear.
 * - `forget` drops a deleted paper everywhere; `clearHistory` empties history only.
 * History beyond MAX_HISTORY is discarded oldest-first.
 */
export function paperSessionsReducer(state: PaperSessions, action: PaperSessionAction): PaperSessions {
  switch (action.type) {
    case "open": {
      const { id } = action;
      if (state.pinned.includes(id)) return { ...state, activeId: id, touched: true };
      const pinned = [...state.pinned, id];
      const shifted = pinned.length > MAX_PINNED ? pinned.splice(0, pinned.length - MAX_PINNED) : [];
      const history = remember(state.history.filter((item) => item !== id), shifted.reverse());
      return { pinned, history, activeId: id, touched: true };
    }
    case "close": {
      const index = state.pinned.indexOf(action.id);
      const pinned = state.pinned.filter((item) => item !== action.id);
      const activeId = state.activeId === action.id
        ? (pinned[Math.min(index, pinned.length - 1)] ?? null)
        : state.activeId;
      return { pinned, history: remember(state.history, [action.id]), activeId, touched: true };
    }
    case "deselect":
      return state.activeId === null ? state : { ...state, activeId: null };
    case "forget":
      return {
        ...state,
        pinned: state.pinned.filter((item) => item !== action.id),
        history: state.history.filter((item) => item !== action.id),
        activeId: state.activeId === action.id ? null : state.activeId,
      };
    case "clearHistory":
      return state.history.length ? { ...state, history: [] } : state;
  }
}

const STORAGE_KEY = "arie.ui.sessions";

/** Restore open paper tabs from this browser; any invalid or unreadable value starts empty. */
export function loadSessions(): PaperSessions {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null") as Partial<PaperSessions> | null;
    const ids = (items: unknown, limit: number) =>
      Array.isArray(items) ? items.filter((item): item is string => typeof item === "string").slice(0, limit) : [];
    if (!value || typeof value !== "object") return EMPTY_SESSIONS;
    return {
      pinned: ids(value.pinned, MAX_PINNED),
      history: ids(value.history, MAX_HISTORY),
      activeId: typeof value.activeId === "string" ? value.activeId : null,
      touched: value.touched === true,
    };
  } catch {
    return EMPTY_SESSIONS;
  }
}

export function saveSessions(sessions: PaperSessions): void {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions)); } catch { /* Tabs still work this visit. */ }
}
