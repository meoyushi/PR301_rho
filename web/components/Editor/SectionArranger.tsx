"use client";
import { useState } from "react";
import { useResumeStore } from "@/lib/resumeStore";

const LABELS: Record<string, string> = {
  summary: "Summary", skills: "Skills", work: "Experience",
  projects: "Projects", achievements: "Achievements", education: "Education",
};

// Drag section cards to reorder the printed sheet; toggle the eye to keep a
// section's data but drop it from the output. Order + visibility flow straight
// into the preview and both exports.
export function SectionArranger() {
  const order = useResumeStore((s) => s.style.sectionOrder);
  const hidden = useResumeStore((s) => s.style.hiddenSections);
  const move = useResumeStore((s) => s.moveSection);
  const toggle = useResumeStore((s) => s.toggleSection);
  const [dragIdx, setDragIdx] = useState<number | null>(null);
  const [overIdx, setOverIdx] = useState<number | null>(null);

  const drop = (to: number) => {
    if (dragIdx !== null && dragIdx !== to) move(dragIdx, to);
    setDragIdx(null); setOverIdx(null);
  };

  return (
    <ul className="space-y-1.5">
      {order.map((key, i) => {
        const off = hidden.includes(key);
        return (
          <li
            key={key}
            draggable
            onDragStart={() => setDragIdx(i)}
            onDragEnter={() => setOverIdx(i)}
            onDragOver={(e) => e.preventDefault()}
            onDrop={() => drop(i)}
            onDragEnd={() => { setDragIdx(null); setOverIdx(null); }}
            className={`flex items-center gap-2 rounded-md border bg-surface-raised px-2.5 py-2 transition-all ${
              dragIdx === i ? "opacity-40" : ""
            } ${overIdx === i && dragIdx !== null && dragIdx !== i ? "border-studio" : "border-hairline"}`}
          >
            <span aria-hidden className="cursor-grab select-none font-label text-ink-muted active:cursor-grabbing">⠿</span>
            <span className="font-label text-[10px] uppercase tracking-[0.12em] text-ink-muted">{String(i + 1).padStart(2, "0")}</span>
            <span className={`flex-1 text-[13px] ${off ? "text-ink-muted line-through" : "text-ink"}`}>{LABELS[key] ?? key}</span>
            <button
              onClick={() => toggle(key)}
              aria-label={off ? `Show ${LABELS[key]}` : `Hide ${LABELS[key]}`}
              aria-pressed={!off}
              className={`rounded p-1 transition-colors ${off ? "text-ink-muted hover:text-ink" : "text-studio hover:opacity-70"}`}
            >
              {off ? <EyeOff /> : <Eye />}
            </button>
          </li>
        );
      })}
    </ul>
  );
}

const Eye = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z" /><circle cx="12" cy="12" r="3" />
  </svg>
);
const EyeOff = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="M9.9 4.24A9.1 9.1 0 0 1 12 4c6.5 0 10 7 10 7a13.2 13.2 0 0 1-1.67 2.68" />
    <path d="M6.6 6.6C3.6 8.3 2 11 2 11s3.5 7 10 7a9 9 0 0 0 5.4-1.6" /><path d="m2 2 20 20" />
  </svg>
);
