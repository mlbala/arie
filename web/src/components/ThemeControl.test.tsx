import { fireEvent, render, screen } from "@testing-library/react";
import { act } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { ThemeControl } from "./ThemeControl";

afterEach(() => { localStorage.removeItem("arie.ui.theme"); delete document.documentElement.dataset.theme; vi.unstubAllGlobals(); });

it("follows system changes, persists an explicit override, and restores it", () => {
  localStorage.removeItem("arie.ui.theme");
  const media = { matches: true, addEventListener: vi.fn(), removeEventListener: vi.fn() };
  vi.stubGlobal("matchMedia", vi.fn(() => media));
  const first = render(<ThemeControl />);
  expect(document.documentElement.dataset.theme).toBe("dark");
  expect(screen.getByRole("button", { name: "System theme" })).toHaveAttribute("aria-pressed", "true");
  media.matches = false;
  act(() => media.addEventListener.mock.calls[0]![1]());
  expect(document.documentElement.dataset.theme).toBe("light");
  fireEvent.click(screen.getByRole("button", { name: "Dark theme" }));
  expect(localStorage.getItem("arie.ui.theme")).toBe("dark");
  media.matches = false;
  act(() => media.addEventListener.mock.calls.at(-1)![1]());
  expect(document.documentElement.dataset.theme).toBe("dark");
  first.unmount();
  render(<ThemeControl />);
  expect(screen.getByRole("button", { name: "Dark theme" })).toHaveAttribute("aria-pressed", "true");
  expect(document.documentElement.dataset.theme).toBe("dark");
});
