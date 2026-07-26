"use client";
import { useState } from "react";
import { BackendUnreachable, parseResume } from "@/lib/api";
import { useResumeStore } from "@/lib/resumeStore";

export function UploadDropzone() {
  const setResume = useResumeStore((s) => s.setResume);
  const setProvenance = useResumeStore((s) => s.setProvenance);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onFile(file: File) {
    setBusy(true); setError(null);
    try {
      const res = await parseResume(file);
      setResume(res.structured_resume);
      setProvenance(res.provenance_map);
    } catch (e) {
      setError(e instanceof BackendUnreachable
        ? "Backend unreachable. Start it: uvicorn rho.api.app:app --reload"
        : `Parse failed: ${(e as Error).message}`);
    } finally { setBusy(false); }
  }

  return (
    <div>
      <label
        className="group block cursor-pointer rounded-sm border border-dashed border-ink-muted/40 bg-desk/60 px-6 py-8 text-center transition-colors hover:border-studio hover:bg-studio/5"
      >
        <input type="file" accept=".pdf,.docx,.txt" className="hidden"
          onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
        <span className="block font-label text-[11px] uppercase tracking-[0.18em] text-ink-muted group-hover:text-studio">
          {busy ? "Reading…" : "Drop or select a file"}
        </span>
        <span className="mt-1 block text-sm text-ink">
          {busy ? "Parsing your résumé" : "PDF, DOCX, or TXT"}
        </span>
      </label>
      {error && (
        <p className="mt-2 border-l-2 border-studio pl-2 text-sm text-studio">{error}</p>
      )}
    </div>
  );
}
