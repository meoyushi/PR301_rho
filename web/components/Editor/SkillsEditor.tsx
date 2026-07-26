"use client";
import { useState } from "react";
import { useResumeStore } from "@/lib/resumeStore";

export function SkillsEditor() {
  const resume = useResumeStore((s) => s.resume);
  const { addSkill, removeSkill } = useResumeStore.getState();
  const [draft, setDraft] = useState("");
  if (!resume) return null;
  return (
    <div className="space-y-3">
      <h2 className="font-label text-[11px] uppercase tracking-[0.18em] text-ink-muted">Skills</h2>
      <div className="flex flex-wrap gap-1.5">
        {resume.skills.map((s) => (
          <span key={s} className="group flex items-center gap-1 rounded-full border border-hairline bg-surface-raised py-0.5 pl-2.5 pr-1.5 text-sm text-ink">
            {s}
            <button
              className="text-ink-muted transition-colors group-hover:text-studio"
              aria-label={`Remove ${s}`}
              onClick={() => removeSkill(s)}>×</button>
          </span>
        ))}
      </div>
      <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); if (draft.trim()) { addSkill(draft.trim()); setDraft(""); } }}>
        <input
          className="flex-1 border-b border-hairline bg-transparent pb-1 text-sm text-ink outline-none transition-colors placeholder:text-ink-muted/50 focus:border-studio"
          value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="Add a skill" />
        <button className="font-label text-[11px] uppercase tracking-[0.1em] text-studio hover:underline" type="submit">Add</button>
      </form>
    </div>
  );
}
