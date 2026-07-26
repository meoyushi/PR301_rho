"use client";
import { useResumeStore } from "@/lib/resumeStore";

// Standalone accomplishments, awards, honors — one line each, freeform text
// (unlike skills, which are short tokens).
export function AchievementsEditor() {
  const resume = useResumeStore((s) => s.resume);
  const { addAchievement, editAchievement, removeAchievement } = useResumeStore.getState();
  if (!resume) return null;
  const items = resume.achievements ?? [];
  return (
    <div className="space-y-3">
      <h2 className="font-label text-[11px] uppercase tracking-[0.18em] text-ink-muted">Achievements</h2>
      {items.length === 0 && (
        <p className="text-xs text-ink-muted">Awards, honors, or recognition not tied to one job.</p>
      )}
      <ul className="space-y-1.5">
        {items.map((a, i) => (
          <li key={i} className="flex items-start gap-1.5">
            <span aria-hidden className="mt-2 text-studio">•</span>
            <textarea
              rows={1}
              className="w-full resize-none rounded-sm border border-transparent bg-transparent p-1 text-sm leading-snug text-ink outline-none transition-colors hover:border-hairline focus:border-studio focus:bg-surface-raised"
              value={a}
              onChange={(e) => editAchievement(i, e.target.value)}
              placeholder="e.g. Winner, ACM ICPC Regionals 2021" />
            <button
              className="mt-1 shrink-0 text-ink-muted transition-colors hover:text-studio"
              aria-label={`Remove achievement ${i + 1}`}
              onClick={() => removeAchievement(i)}>×</button>
          </li>
        ))}
      </ul>
      <button
        className="font-label text-[11px] uppercase tracking-[0.1em] text-studio hover:underline"
        onClick={() => addAchievement()}>+ Achievement</button>
    </div>
  );
}
