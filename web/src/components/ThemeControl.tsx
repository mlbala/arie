import { Monitor, Moon, Sun } from "lucide-react";
import { useEffect, useState } from "react";

type Theme = "light" | "dark" | "system";
function savedTheme(): Theme {
  try {
    const value = localStorage.getItem("arie.ui.theme");
    return value === "dark" || value === "light" ? value : "system";
  } catch { return "system"; }
}

/** Persist the chosen theme and react to operating-system changes in System mode. */
export function ThemeControl() {
  const [theme, setTheme] = useState<Theme>(savedTheme);
  useEffect(() => {
    const media = window.matchMedia?.("(prefers-color-scheme: dark)");
    const apply = () => {
      document.documentElement.dataset.theme = theme === "system" ? media?.matches ? "dark" : "light" : theme;
    };
    apply();
    try { localStorage.setItem("arie.ui.theme", theme); } catch { /* Theme still applies to this session. */ }
    media?.addEventListener("change", apply);
    return () => media?.removeEventListener("change", apply);
  }, [theme]);
  return <div className="theme-control" role="group" aria-label="Appearance">
    {([["light", Sun, "Light"], ["dark", Moon, "Dark"], ["system", Monitor, "System"]] as const).map(([value, Icon, label]) =>
      <button key={value} aria-label={`${label} theme`} aria-pressed={theme === value} title={`${label} theme`} onClick={() => setTheme(value)}><Icon size={13} /><span>{label}</span></button>,
    )}
  </div>;
}
