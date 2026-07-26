import { create } from "zustand";
import type { StructuredResume } from "./types";
import { TEMPLATES, type TemplateId } from "./templates";

export type SectionKey = "summary" | "skills" | "work" | "projects" | "education";
export type Theme = "light" | "dark";

export interface StyleSettings {
  fontSize: number; margin: number; lineSpacing: number; accent: string;
  sectionOrder: string[];
  template: TemplateId;
  hiddenSections: string[];   // sections toggled off in the output
}
export interface OptimizeView {
  score: number; previousScore: number | null;
  baselineScore: number | null; // this run's ORIGINAL-résumé score (before tailoring)
  components: { label: string; before: number; after: number }[];
  gaps: { text: string; priority: string; status: string }[];
  fabricationsBlocked: number;
  originalResume: StructuredResume; // pre-optimize, for before/after
}

interface State {
  resume: StructuredResume | null;
  provenance: unknown;
  style: StyleSettings;
  optimize: OptimizeView | null;
  theme: Theme;
  sidebarWidth: number;
  sidebarCollapsed: boolean;
  setResume: (r: StructuredResume) => void;
  setProvenance: (p: unknown) => void;
  setField: <K extends keyof StructuredResume>(k: K, v: StructuredResume[K]) => void;
  setTemplate: (id: TemplateId) => void;
  moveSection: (from: number, to: number) => void;
  toggleSection: (key: string) => void;
  setTheme: (t: Theme) => void;
  toggleTheme: () => void;
  setSidebarWidth: (w: number) => void;
  toggleSidebar: () => void;
  addBullet: (workIdx: number) => void;
  editBullet: (workIdx: number, bulletIdx: number, text: string) => void;
  removeBullet: (workIdx: number, bulletIdx: number) => void;
  addSkill: (s: string) => void;
  removeSkill: (s: string) => void;
  addAchievement: () => void;
  editAchievement: (idx: number, text: string) => void;
  removeAchievement: (idx: number) => void;
  addProjectBullet: (projIdx: number) => void;
  editProjectBullet: (projIdx: number, bulletIdx: number, text: string) => void;
  removeProjectBullet: (projIdx: number, bulletIdx: number) => void;
  setStyle: (patch: Partial<StyleSettings>) => void;
  applyOptimize: (view: {
    tailored: StructuredResume;
    displayScore: number;
    baselineDisplayScore: number | null;
    components: OptimizeView["components"];
    gaps: OptimizeView["gaps"];
    fabricationsBlocked: number;
    previousScore: number | null;
  }) => void;
}

const DEFAULT_STYLE: StyleSettings = {
  fontSize: 14, margin: 48, lineSpacing: 1.4, accent: "#b5482a",
  sectionOrder: ["summary", "skills", "work", "projects", "achievements", "education"],
  template: "classic",
  hiddenSections: [],
};

const SIDEBAR_MIN = 320;
const SIDEBAR_MAX = 620;
const SIDEBAR_DEFAULT = 416;
const clampWidth = (w: number) => Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, Math.round(w)));

// ── Persistence (client only) ────────────────────────────────────────────────
// The résumé itself is not persisted (privacy: it never touches localStorage);
// only the workbench preferences — style, theme, sidebar geometry — are.
const LS_KEY = "rho-prefs";
interface Prefs { style: StyleSettings; theme: Theme; sidebarWidth: number; sidebarCollapsed: boolean; }

function loadPrefs(): Partial<Prefs> {
  if (typeof window === "undefined") return {};
  try {
    const raw = localStorage.getItem(LS_KEY);
    return raw ? (JSON.parse(raw) as Partial<Prefs>) : {};
  } catch { return {}; }
}

function savePrefs(p: Prefs) {
  if (typeof window === "undefined") return;
  try { localStorage.setItem(LS_KEY, JSON.stringify(p)); } catch { /* quota / disabled */ }
}

// Merge saved style over defaults, then repair the section order: a persisted
// order from before a section existed (e.g. "achievements") would silently drop
// that section, so append any default section the saved order is missing.
function mergeStyle(saved?: StyleSettings): StyleSettings {
  const style = { ...DEFAULT_STYLE, ...saved };
  const order = [...style.sectionOrder];
  for (const key of DEFAULT_STYLE.sectionOrder) {
    if (!order.includes(key)) order.push(key);
  }
  return { ...style, sectionOrder: order };
}

function initialTheme(saved: Partial<Prefs>): Theme {
  if (saved.theme) return saved.theme;
  if (typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: dark)").matches) return "dark";
  return "light";
}

function mutate(r: StructuredResume, fn: (draft: StructuredResume) => void): StructuredResume {
  const copy: StructuredResume = JSON.parse(JSON.stringify(r));
  fn(copy);
  return copy;
}

// Backfill array fields a résumé may be missing (e.g. one parsed before the
// projects field existed), so every component can read them without guarding.
function normalize(r: StructuredResume): StructuredResume {
  return {
    ...r,
    work: r.work ?? [],
    education: r.education ?? [],
    projects: r.projects ?? [],
    skills: r.skills ?? [],
    certifications: r.certifications ?? [],
    achievements: r.achievements ?? [],
    emails: r.emails ?? [],
    phones: r.phones ?? [],
    urls: r.urls ?? [],
  };
}

const _saved = loadPrefs();

function applyTheme(t: Theme) {
  if (typeof document !== "undefined") document.documentElement.setAttribute("data-theme", t);
}

export const useResumeStore = create<State>((set, get) => {
  const persist = () => {
    const s = get();
    savePrefs({ style: s.style, theme: s.theme, sidebarWidth: s.sidebarWidth, sidebarCollapsed: s.sidebarCollapsed });
  };
  return {
  resume: null, provenance: null,
  style: mergeStyle(_saved.style),
  optimize: null,
  theme: initialTheme(_saved),
  sidebarWidth: clampWidth(_saved.sidebarWidth ?? SIDEBAR_DEFAULT),
  sidebarCollapsed: _saved.sidebarCollapsed ?? false,
  setResume: (r) => set({ resume: normalize(r), optimize: null }),
  setProvenance: (p) => set({ provenance: p }),
  setField: (k, v) => set((s) => ({ resume: s.resume && mutate(s.resume, (d) => { (d as any)[k] = v; }) })),
  setTemplate: (id) => { set((s) => ({ style: { ...s.style, template: id, accent: TEMPLATES[id].accent } })); persist(); },
  moveSection: (from, to) => { set((s) => {
    const order = [...s.style.sectionOrder];
    if (from < 0 || from >= order.length || to < 0 || to >= order.length) return s;
    const [item] = order.splice(from, 1);
    order.splice(to, 0, item);
    return { style: { ...s.style, sectionOrder: order } };
  }); persist(); },
  toggleSection: (key) => { set((s) => {
    const hidden = s.style.hiddenSections.includes(key)
      ? s.style.hiddenSections.filter((k) => k !== key)
      : [...s.style.hiddenSections, key];
    return { style: { ...s.style, hiddenSections: hidden } };
  }); persist(); },
  setTheme: (t) => { applyTheme(t); set({ theme: t }); persist(); },
  toggleTheme: () => { const t = get().theme === "dark" ? "light" : "dark"; applyTheme(t); set({ theme: t }); persist(); },
  setSidebarWidth: (w) => { set({ sidebarWidth: clampWidth(w) }); persist(); },
  toggleSidebar: () => { set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })); persist(); },
  addBullet: (wi) => set((s) => ({ resume: s.resume && mutate(s.resume, (d) => { d.work[wi].bullets.push(""); }) })),
  editBullet: (wi, bi, text) => set((s) => ({ resume: s.resume && mutate(s.resume, (d) => { d.work[wi].bullets[bi] = text; }) })),
  removeBullet: (wi, bi) => set((s) => ({ resume: s.resume && mutate(s.resume, (d) => { d.work[wi].bullets.splice(bi, 1); }) })),
  addSkill: (skill) => set((s) => ({ resume: s.resume && mutate(s.resume, (d) => { if (!d.skills.map((x) => x.toLowerCase()).includes(skill.toLowerCase())) d.skills.push(skill); }) })),
  removeSkill: (skill) => set((s) => ({ resume: s.resume && mutate(s.resume, (d) => { d.skills = d.skills.filter((x) => x.toLowerCase() !== skill.toLowerCase()); }) })),
  addAchievement: () => set((s) => ({ resume: s.resume && mutate(s.resume, (d) => { d.achievements = [...(d.achievements ?? []), ""]; }) })),
  editAchievement: (i, text) => set((s) => ({ resume: s.resume && mutate(s.resume, (d) => { d.achievements[i] = text; }) })),
  removeAchievement: (i) => set((s) => ({ resume: s.resume && mutate(s.resume, (d) => { d.achievements.splice(i, 1); }) })),
  addProjectBullet: (pi) => set((s) => ({ resume: s.resume && mutate(s.resume, (d) => { d.projects[pi].bullets.push(""); }) })),
  editProjectBullet: (pi, bi, text) => set((s) => ({ resume: s.resume && mutate(s.resume, (d) => { d.projects[pi].bullets[bi] = text; }) })),
  removeProjectBullet: (pi, bi) => set((s) => ({ resume: s.resume && mutate(s.resume, (d) => { d.projects[pi].bullets.splice(bi, 1); }) })),
  setStyle: (patch) => { set((s) => ({ style: { ...s.style, ...patch } })); persist(); },
  applyOptimize: (v) => set((s) => ({
    optimize: {
      score: v.displayScore,
      previousScore: v.previousScore,
      baselineScore: v.baselineDisplayScore,
      components: v.components,
      gaps: v.gaps,
      fabricationsBlocked: v.fabricationsBlocked,
      originalResume: s.resume!, // the résumé that went in, for before/after
    },
    resume: normalize(v.tailored),
  })),
  };
});
