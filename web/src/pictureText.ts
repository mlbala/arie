// pymupdf4llm wraps the words drawn inside a figure in these markers. They stay in the stored
// page text, and every offset counts them; only what is displayed leaves them out.
const MARKERS = /<!-- (?:Start|End) of picture text -->/g;

/** `text` as shown to a reader: the picture-text markers removed, nothing else changed. */
export function withoutPictureMarkers(text: string): string {
  return text.replace(MARKERS, "");
}
