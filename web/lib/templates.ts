// Résumé templates = paper-stock samples. Each one controls the printed sheet's
// layout, typography, heading treatment, and default accent. The editor chrome
// (the "desk") is unaffected — templates only restyle the sheet.

export type TemplateId = "classic" | "modern" | "compact" | "bold";
export type Layout = "single" | "two-column";
export type HeadingStyle = "underline" | "bar" | "smallcaps" | "block";

export interface Template {
  id: TemplateId;
  name: string;
  blurb: string;
  layout: Layout;
  heading: HeadingStyle;
  accent: string;          // sensible default; user can still override
  bodyFont: string;        // CSS font-family for the sheet body
  displayFont: string;     // CSS font-family for the name/headings
  nameAlign: "left" | "center";
  scale: number;           // relative density multiplier for spacing
}

const SERIF = 'ui-serif, Georgia, "Times New Roman", serif';
const SANS = 'ui-sans-serif, "Inter", system-ui, -apple-system, sans-serif';

export const TEMPLATES: Record<TemplateId, Template> = {
  classic: {
    id: "classic",
    name: "Classic",
    blurb: "Serif, centered, timeless",
    layout: "single",
    heading: "underline",
    accent: "#b5482a",
    bodyFont: SERIF,
    displayFont: SERIF,
    nameAlign: "center",
    scale: 1,
  },
  modern: {
    id: "modern",
    name: "Modern",
    blurb: "Two-column, sans, left rail",
    layout: "two-column",
    heading: "bar",
    accent: "#2f5d8a",
    bodyFont: SANS,
    displayFont: SANS,
    nameAlign: "left",
    scale: 1,
  },
  compact: {
    id: "compact",
    name: "Compact",
    blurb: "Dense, small caps, one page",
    layout: "single",
    heading: "smallcaps",
    accent: "#1c1b19",
    bodyFont: SANS,
    displayFont: SANS,
    nameAlign: "left",
    scale: 0.82,
  },
  bold: {
    id: "bold",
    name: "Bold",
    blurb: "Big serif name, block headings",
    layout: "single",
    heading: "block",
    accent: "#a23b1e",
    bodyFont: SANS,
    displayFont: SERIF,
    nameAlign: "left",
    scale: 1.06,
  },
};

export const TEMPLATE_LIST: Template[] = Object.values(TEMPLATES);
