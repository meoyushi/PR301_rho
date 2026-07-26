import type { Config } from "tailwindcss";
export default {
  darkMode: ["selector", '[data-theme="dark"]'],
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "var(--ink)",
        "ink-muted": "var(--ink-muted)",
        paper: "var(--paper)",
        desk: "var(--desk)",
        surface: "var(--surface)",
        "surface-raised": "var(--surface-raised)",
        hairline: "var(--hairline)",
        studio: "var(--studio)",
        "studio-soft": "var(--studio-soft)",
      },
      fontFamily: {
        tool: ["var(--font-tool)"],
        label: ["var(--font-label)"],
        sheet: ["var(--font-sheet)"],
      },
      boxShadow: {
        sheet: "var(--shadow-sheet)",
      },
    },
  },
  plugins: [],
} satisfies Config;
