"use client";
import { useResumeStore } from "@/lib/resumeStore";

export function FieldEditors() {
  const resume = useResumeStore((s) => s.resume);
  const setField = useResumeStore((s) => s.setField);
  if (!resume) return null;
  return (
    <div className="space-y-3">
      <h2 className="font-label text-[11px] uppercase tracking-[0.18em] text-ink-muted">Identity</h2>
      <div className="space-y-2">
        <input
          className="w-full border-b border-hairline bg-transparent pb-1 text-lg font-medium text-ink outline-none transition-colors placeholder:text-ink-muted/50 focus:border-studio"
          value={resume.name} onChange={(e) => setField("name", e.target.value)} placeholder="Name" />
        <input
          className="w-full border-b border-hairline bg-transparent pb-1 text-sm text-ink outline-none transition-colors placeholder:text-ink-muted/50 focus:border-studio"
          value={resume.headline ?? ""} onChange={(e) => setField("headline", e.target.value)} placeholder="Headline" />
        <textarea
          className="min-h-[4.5rem] w-full resize-y rounded-sm border border-hairline bg-surface-raised p-2 text-sm text-ink outline-none transition-colors placeholder:text-ink-muted/50 focus:border-studio"
          value={resume.summary ?? ""} onChange={(e) => setField("summary", e.target.value)} placeholder="Summary" />
      </div>
    </div>
  );
}
