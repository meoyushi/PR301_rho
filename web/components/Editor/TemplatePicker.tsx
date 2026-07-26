"use client";
import { useResumeStore } from "@/lib/resumeStore";
import { TEMPLATE_LIST } from "@/lib/templates";

// Paper-stock samples. Each swatch previews the template's layout as a tiny
// abstract sheet — pick one like choosing letterhead.
export function TemplatePicker() {
  const current = useResumeStore((s) => s.style.template);
  const setTemplate = useResumeStore((s) => s.setTemplate);

  return (
    <div className="grid grid-cols-2 gap-2.5">
      {TEMPLATE_LIST.map((t) => {
        const active = t.id === current;
        return (
          <button
            key={t.id}
            onClick={() => setTemplate(t.id)}
            aria-pressed={active}
            className={`group rounded-md border p-2 text-left transition-all ${
              active
                ? "border-studio bg-studio-soft shadow-[0_0_0_1px_var(--studio)]"
                : "border-hairline bg-surface-raised hover:border-ink-muted"
            }`}
          >
            <MiniSheet template={t} />
            <div className="mt-1.5 flex items-baseline justify-between">
              <span className="text-[13px] font-semibold text-ink">{t.name}</span>
              {active && <span className="font-label text-[9px] uppercase tracking-widest text-studio">on</span>}
            </div>
            <p className="font-label text-[9px] uppercase tracking-[0.08em] text-ink-muted">{t.blurb}</p>
          </button>
        );
      })}
    </div>
  );
}

// A 3:4 abstract résumé thumbnail that mirrors the template's real layout.
function MiniSheet({ template }: { template: (typeof TEMPLATE_LIST)[number] }) {
  const a = template.accent;
  const bar = (w: string, key: number, color = "var(--sheet-rule)") => (
    <span key={key} className="block h-[3px] rounded-full" style={{ width: w, background: color }} />
  );
  const twoCol = template.layout === "two-column";
  return (
    <div className="aspect-[3/4] w-full overflow-hidden rounded-sm bg-white p-2 shadow-inner ring-1 ring-black/5">
      {/* name */}
      <div className={`flex flex-col gap-[3px] ${template.nameAlign === "center" ? "items-center" : "items-start"}`}>
        <span className="block h-[5px] rounded-full" style={{ width: "58%", background: a }} />
        {bar("36%", 0)}
      </div>
      {twoCol ? (
        <div className="mt-2 flex gap-2">
          <div className="flex w-1/3 flex-col gap-[3px]">
            <span className="block h-[3px] w-2/3 rounded-full" style={{ background: a }} />
            {bar("90%", 1)}{bar("80%", 2)}{bar("90%", 3)}
          </div>
          <div className="flex flex-1 flex-col gap-[3px]">
            <span className="block h-[3px] w-1/2 rounded-full" style={{ background: a }} />
            {bar("100%", 4)}{bar("95%", 5)}{bar("100%", 6)}{bar("70%", 7)}
          </div>
        </div>
      ) : (
        <div className="mt-2 flex flex-col gap-[3px]">
          <span
            className="block h-[3px] w-1/3 rounded-full"
            style={{ background: a, boxShadow: template.heading === "block" ? `0 0 0 2px ${a}` : undefined }}
          />
          {bar("100%", 1)}{bar("92%", 2)}{bar("97%", 3)}
          <span className="mt-1 block h-[3px] w-1/4 rounded-full" style={{ background: a }} />
          {bar("100%", 4)}{bar("88%", 5)}
        </div>
      )}
    </div>
  );
}
