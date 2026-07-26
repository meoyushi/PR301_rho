"use client";
import { useEffect } from "react";
import { useResumeStore } from "@/lib/resumeStore";

// Dim the workshop. The desk (editor chrome) follows the theme; the sheet stays
// paper in both. Choice persists.
export function ThemeToggle() {
  const theme = useResumeStore((s) => s.theme);
  const toggle = useResumeStore((s) => s.toggleTheme);
  const dark = theme === "dark";

  // Keep the <html data-theme> in lockstep with the store on mount and on every
  // change, so the pre-paint inline script and the store can never drift apart.
  useEffect(() => { document.documentElement.setAttribute("data-theme", theme); }, [theme]);
  return (
    <button
      onClick={toggle}
      role="switch"
      aria-checked={dark}
      aria-label="Toggle dark mode"
      title={dark ? "Switch to light" : "Switch to dark"}
      className="group flex h-8 items-center gap-2 rounded-full border border-hairline bg-surface-raised px-2.5 text-ink-muted transition-colors hover:border-studio hover:text-studio"
    >
      <span aria-hidden className="text-[13px]">{dark ? "☾" : "☀"}</span>
      <span className="font-label text-[10px] uppercase tracking-[0.14em]">{dark ? "Dark" : "Light"}</span>
    </button>
  );
}
