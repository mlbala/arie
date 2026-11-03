import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PublicationWindowFields } from "./PublicationWindow";
import type { PublicationWindow } from "../publication";

function renderFields(value: PublicationWindow, allowAll = false) {
  const onChange = vi.fn<(window: PublicationWindow) => void>();
  render(<PublicationWindowFields value={value} onChange={onChange} allowAll={allowAll} />);
  return onChange;
}

describe("PublicationWindowFields", () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date("2026-11-03T12:00:00Z"));
  });
  afterEach(() => vi.useRealTimers());

  it("switching mode seeds the current UTC year, month and range", () => {
    const onChange = renderFields({ mode: "relative", initial_months: 6 });
    const mode = screen.getByLabelText("Publication window");
    fireEvent.change(mode, { target: { value: "month" } });
    expect(onChange).toHaveBeenLastCalledWith({ mode: "month", initial_months: 6, year: 2026, month: 11 });
    fireEvent.change(mode, { target: { value: "range" } });
    expect(onChange).toHaveBeenLastCalledWith({ mode: "range", initial_months: 6, start_month: "2026-11", end_month: "2026-11" });
  });

  it("keeps a month range ordered when the start moves past the end", () => {
    const onChange = renderFields({ mode: "range", initial_months: 6, start_month: "2025-01", end_month: "2025-03" });
    fireEvent.change(screen.getByLabelText("From month month"), { target: { value: "6" } });
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ start_month: "2025-06", end_month: "2025-06" }));
  });

  it("offers all dates only where allowed and stops the year picker at this year", () => {
    renderFields({ mode: "relative", initial_months: 6 });
    expect(screen.queryByRole("option", { name: "All dates" })).not.toBeInTheDocument();
    cleanup();
    renderFields({ mode: "month", initial_months: 6, year: 2026, month: 11 }, true);
    expect(screen.getAllByRole("option", { name: "All dates" }).length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "Next year" })).toBeDisabled();
    expect(screen.getByText("November 2026")).toBeInTheDocument();
  });
});

