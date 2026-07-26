"use client";
import { useState } from "react";
import { downloadDocx, BackendUnreachable } from "@/lib/api";
import { useResumeStore } from "@/lib/resumeStore";

export function DownloadBar() {
  const resume = useResumeStore((s) => s.resume);
  const style = useResumeStore((s) => s.style);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (!resume) return null;

  const filename = (resume.name || "resume").trim().replace(/\s+/g, "_");

  async function docx() {
    if (!resume) return;
    setBusy(true); setError(null);
    try {
      const blob = await downloadDocx(resume, style.sectionOrder, style.accent, style.hiddenSections);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = `${filename}.docx`;
      document.body.appendChild(a); a.click(); a.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(e instanceof BackendUnreachable
        ? "Backend unreachable — start it to export DOCX."
        : (e as Error).message);
    } finally { setBusy(false); }
  }

  return (
    <div className="space-y-2 border-t border-hairline pt-4">
      <h2 className="font-label text-[11px] uppercase tracking-[0.18em] text-ink-muted">Download</h2>
      <div className="flex gap-2">
        <button
          className="flex-1 rounded-sm border border-hairline bg-surface-raised py-2 font-label text-[11px] uppercase tracking-[0.14em] text-ink transition-colors hover:border-studio hover:text-studio"
          onClick={() => window.print()}>
          PDF
        </button>
        <button disabled={busy}
          className="flex-1 rounded-sm border border-hairline bg-surface-raised py-2 font-label text-[11px] uppercase tracking-[0.14em] text-ink transition-colors hover:border-studio hover:text-studio disabled:cursor-not-allowed disabled:text-ink-muted"
          onClick={docx}>
          {busy ? "…" : "DOCX"}
        </button>
      </div>
      <p className="text-xs text-ink-muted">PDF prints the preview exactly. DOCX is an ATS-friendly Word file.</p>
      {error && <p className="border-l-2 border-studio pl-2 text-sm text-studio">{error}</p>}
    </div>
  );
}
