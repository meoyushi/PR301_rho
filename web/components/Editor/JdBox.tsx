"use client";
import { useState } from "react";
import { startOptimize, pollOptimize, BackendUnreachable } from "@/lib/api";
import { useResumeStore } from "@/lib/resumeStore";

export function JdBox() {
  const resume = useResumeStore((s) => s.resume);
  const optimize = useResumeStore((s) => s.optimize);
  const { applyOptimize } = useResumeStore.getState();
  const [jd, setJd] = useState("");
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    if (!resume || !jd.trim()) return;
    setBusy(true); setError(null); setStage("starting");
    const prevScore = optimize?.score ?? null;
    try {
      const started = await startOptimize(resume, jd);
      const done = await pollOptimize(started.id, {
        intervalMs: 1500, timeoutMs: 180000,
      });
      const r = done.result!;
      applyOptimize({
        tailored: r.tailored_resume.resume,
        // Coalesce so a missing field from an older backend cannot make the
        // score undefined and crash the render.
        displayScore: r.display_score ?? r.final_score ?? 0,
        baselineDisplayScore: r.baseline_display_score ?? null,
        components: r.components ?? [],
        gaps: r.match_result.gaps.map((g) => ({ text: g.requirement.text, priority: g.requirement.priority, status: g.status })),
        fabricationsBlocked: r.tailored_resume.fabrication_report.rejected_edits.length,
        previousScore: prevScore,
      });
    } catch (e) {
      setError(e instanceof BackendUnreachable
        ? "Backend unreachable. Start it: uvicorn rho.api.app:app --reload"
        : (e as Error).message);
    } finally { setBusy(false); setStage(null); }
  }

  return (
    <div className="space-y-3 border-t border-hairline pt-4">
      <h2 className="font-label text-[11px] uppercase tracking-[0.18em] text-ink-muted">Target role</h2>
      <textarea
        className="h-32 w-full resize-y rounded-sm border border-hairline bg-surface-raised p-2 text-sm text-ink outline-none transition-colors placeholder:text-ink-muted/50 focus:border-studio"
        value={jd} onChange={(e) => setJd(e.target.value)} placeholder="Paste the target job description…" />
      <button disabled={!resume || !jd.trim() || busy}
        className="w-full rounded-sm bg-ink py-2.5 font-label text-[11px] uppercase tracking-[0.14em] text-paper transition-colors hover:bg-studio disabled:cursor-not-allowed disabled:bg-ink-muted/30 disabled:text-ink-muted"
        onClick={run}>
        {busy ? `Optimising… ${stage ?? ""}` : "Optimise score →"}
      </button>
      {error && (
        <p className="border-l-2 border-studio pl-2 text-sm text-studio">{error}</p>
      )}
      {optimize && !busy && typeof optimize.score === "number" && (
        <div className="space-y-3 rounded-sm border border-hairline bg-surface-raised p-3">
          <div className="flex items-baseline gap-2">
            <span className="font-label text-[11px] uppercase tracking-[0.1em] text-ink-muted">Match score</span>
            <span className="text-2xl font-semibold text-ink">{optimize.score.toFixed(0)}</span>
            <span className="text-sm text-ink-muted">/100</span>
            {typeof optimize.baselineScore === "number" && (
              <span className={`ml-1 rounded-sm px-1.5 py-0.5 font-label text-[11px] ${
                optimize.score > optimize.baselineScore
                  ? "bg-studio/10 text-studio"
                  : optimize.score < optimize.baselineScore
                  ? "bg-orange-100 text-orange-700"
                  : "bg-ink-muted/10 text-ink-muted"
              }`}>
                {optimize.score > optimize.baselineScore ? "▲" : optimize.score < optimize.baselineScore ? "▼" : "="}{" "}
                {(optimize.score - optimize.baselineScore >= 0 ? "+" : "")}
                {(optimize.score - optimize.baselineScore).toFixed(0)} from {optimize.baselineScore.toFixed(0)}
              </span>
            )}
          </div>

          {(optimize.components?.length ?? 0) > 0 && (
            <div className="space-y-1.5 border-t border-hairline pt-2">
              <span className="font-label text-[11px] uppercase tracking-[0.1em] text-ink-muted">Improvement breakdown</span>
              {optimize.components.map((c) => {
                const beforePct = Math.round(c.before * 100);
                const afterPct = Math.round(c.after * 100);
                const delta = afterPct - beforePct;
                return (
                  <div key={c.label} className="flex items-center gap-2 text-sm">
                    <span className="w-40 shrink-0 text-ink-muted">{c.label}</span>
                    <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-ink-muted/10">
                      <div className="absolute inset-y-0 left-0 rounded-full bg-ink-muted/30" style={{ width: `${beforePct}%` }} />
                      <div className="absolute inset-y-0 left-0 rounded-full bg-studio transition-all" style={{ width: `${afterPct}%` }} />
                    </div>
                    <span className="w-24 shrink-0 text-right tabular-nums text-ink">
                      {beforePct}% → {afterPct}%
                      {delta !== 0 && (
                        <span className={delta > 0 ? "ml-1 text-studio" : "ml-1 text-orange-700"}>
                          ({delta > 0 ? "+" : ""}{delta}%)
                        </span>
                      )}
                    </span>
                  </div>
                );
              })}
            </div>
          )}

          <div className="border-t border-hairline pt-2 text-sm text-ink-muted">
            Unsourced edits blocked: <span className="text-ink">{optimize.fabricationsBlocked}</span>
            <span className="ml-1 text-xs">(fabrication gate)</span>
          </div>
          {(optimize.gaps?.length ?? 0) > 0 && <KeywordCoverage gaps={optimize.gaps} />}
        </div>
      )}
    </div>
  );
}

// ATS keyword coverage: every requirement pulled from the JD, chipped by whether
// the résumé already covers it. Green = present, amber = weak/partial, red =
// missing. A recruiter's ATS scores you on exactly these — this shows the gaps
// to close before you export.
function KeywordCoverage({ gaps }: { gaps: { text: string; priority: string; status: string }[] }) {
  const order = { absent: 0, weak: 1, present: 2 } as const;
  const sorted = [...gaps].sort(
    (a, b) => (order[a.status as keyof typeof order] ?? 3) - (order[b.status as keyof typeof order] ?? 3),
  );
  const covered = gaps.filter((g) => g.status === "present").length;
  const style: Record<string, string> = {
    present: "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300",
    weak: "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300",
    absent: "border-studio/40 bg-studio-soft text-studio",
  };
  const dot: Record<string, string> = { present: "bg-emerald-500", weak: "bg-amber-500", absent: "bg-studio" };
  return (
    <div className="space-y-2 border-t border-hairline pt-2">
      <div className="flex items-baseline justify-between">
        <span className="font-label text-[11px] uppercase tracking-[0.1em] text-ink-muted">Keyword coverage</span>
        <span className="font-label text-[11px] tabular-nums text-ink-muted">{covered}/{gaps.length} matched</span>
      </div>
      <ul className="flex flex-wrap gap-1.5">
        {sorted.map((g, i) => (
          <li key={i}
            title={`${g.status} · ${g.priority} priority`}
            className={`flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs ${style[g.status] ?? style.absent}`}>
            <span aria-hidden className={`h-1.5 w-1.5 rounded-full ${dot[g.status] ?? dot.absent}`} />
            {g.text}
          </li>
        ))}
      </ul>
    </div>
  );
}
