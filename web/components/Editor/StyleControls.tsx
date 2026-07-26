"use client";
import { useResumeStore } from "@/lib/resumeStore";

export function StyleControls() {
  const style = useResumeStore((s) => s.style);
  const setStyle = useResumeStore((s) => s.setStyle);
  return (
    <div className="space-y-3 pt-1">
      <label className="block">
        <span className="flex items-baseline justify-between font-label text-[11px] uppercase tracking-[0.1em] text-ink-muted">
          Font size <span className="text-ink">{style.fontSize}px</span>
        </span>
        <input type="range" min={10} max={20} value={style.fontSize}
          onChange={(e) => setStyle({ fontSize: +e.target.value })}
          className="mt-1.5 w-full accent-studio" />
      </label>
      <label className="block">
        <span className="flex items-baseline justify-between font-label text-[11px] uppercase tracking-[0.1em] text-ink-muted">
          Margin <span className="text-ink">{style.margin}px</span>
        </span>
        <input type="range" min={16} max={80} value={style.margin}
          onChange={(e) => setStyle({ margin: +e.target.value })}
          className="mt-1.5 w-full accent-studio" />
      </label>
      <label className="block">
        <span className="flex items-baseline justify-between font-label text-[11px] uppercase tracking-[0.1em] text-ink-muted">
          Line spacing <span className="text-ink">{style.lineSpacing}</span>
        </span>
        <input type="range" min={1} max={2} step={0.1} value={style.lineSpacing}
          onChange={(e) => setStyle({ lineSpacing: +e.target.value })}
          className="mt-1.5 w-full accent-studio" />
      </label>
      <label className="flex items-center justify-between">
        <span className="font-label text-[11px] uppercase tracking-[0.1em] text-ink-muted">Accent</span>
        <input type="color" value={style.accent} onChange={(e) => setStyle({ accent: e.target.value })}
          className="h-6 w-10 cursor-pointer rounded-sm border border-hairline bg-transparent" />
      </label>
    </div>
  );
}
