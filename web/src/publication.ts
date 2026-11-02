import type { SyncSettings } from "./api/client";

export type PublicationWindow = Pick<SyncSettings, "initial_months" | "year" | "month" | "start_month" | "end_month"> & {
  mode: SyncSettings["mode"] | "all";
};
export const RECENT_WINDOW: PublicationWindow = { mode: "relative", initial_months: 6 };
export const MONTHS = Array.from({ length: 12 }, (_, month) =>
  new Intl.DateTimeFormat("en", { month: "long", timeZone: "UTC" }).format(new Date(Date.UTC(2000, month, 1))),
);
export const monthKey = (year: number, month: number) => `${year}-${String(month).padStart(2, "0")}`;

export function publicationParams(value: PublicationWindow): Record<string, string> {
  if (value.mode === "relative") return { months: String(value.initial_months) };
  if (value.mode === "year") return { year: String(value.year) };
  if (value.mode === "month") return { month: monthKey(value.year!, value.month!) };
  if (value.mode === "since") return { start_month: value.start_month! };
  if (value.mode === "range") return { start_month: value.start_month!, end_month: value.end_month! };
  return {};
}

export function publicationLabel(value: PublicationWindow): string {
  if (value.mode === "relative") return `Last ${value.initial_months} months`;
  if (value.mode === "year") return `Year ${value.year}`;
  if (value.mode === "month") return `${MONTHS[(value.month ?? 1) - 1]} ${value.year}`;
  if (value.mode === "since") return `From ${value.start_month}`;
  if (value.mode === "range") return `${value.start_month} – ${value.end_month}`;
  return "All publication dates";
}
