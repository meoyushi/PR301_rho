"use client";
import { useResumeStore } from "@/lib/resumeStore";

export function ProjectsEditor() {
  const resume = useResumeStore((s) => s.resume);
  const { addProjectBullet, editProjectBullet, removeProjectBullet } = useResumeStore.getState();
  if (!resume || !resume.projects || resume.projects.length === 0) return null;
  return (
    <div className="space-y-3">
      <h2 className="font-label text-[11px] uppercase tracking-[0.18em] text-ink-muted">Projects</h2>
      <div className="space-y-3">
        {resume.projects.map((p, pi) => (
          <div key={pi} className="rounded-sm border border-hairline bg-surface-raised p-3">
            <div className="flex items-baseline justify-between gap-2">
              <span className="font-medium text-ink">{p.name}</span>
              {p.url && (
                <a href={p.url} target="_blank" rel="noreferrer"
                  className="text-sm text-studio hover:underline">link</a>
              )}
            </div>
            {p.tech.length > 0 && (
              <div className="mt-0.5 text-sm italic text-ink-muted">{p.tech.join(", ")}</div>
            )}
            <ul className="mt-2 space-y-1.5">
              {p.bullets.map((b, bi) => (
                <li key={bi} className="flex items-start gap-1.5">
                  <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-studio/50" />
                  <textarea
                    className="w-full resize-none rounded-sm border border-transparent bg-transparent p-1 text-sm leading-snug text-ink outline-none transition-colors hover:border-hairline focus:border-studio focus:bg-surface-raised"
                    value={b} onChange={(e) => editProjectBullet(pi, bi, e.target.value)} rows={1} />
                  <button
                    className="shrink-0 px-1 text-sm text-ink-muted transition-colors hover:text-studio"
                    aria-label="Remove bullet"
                    onClick={() => removeProjectBullet(pi, bi)}>×</button>
                </li>
              ))}
            </ul>
            <button
              className="mt-2 font-label text-[11px] uppercase tracking-[0.1em] text-studio hover:underline"
              onClick={() => addProjectBullet(pi)}>+ bullet</button>
          </div>
        ))}
      </div>
    </div>
  );
}
