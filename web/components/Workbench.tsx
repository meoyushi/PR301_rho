"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { useResumeStore } from "@/lib/resumeStore";

// The drafting table: a resizable, collapsible desk (the editor) beside the
// sheet (the preview). Drag the divider to resize; collapse it for a full-width
// sheet. Width and collapsed state persist. On small screens it stacks and the
// resize handle is hidden — width only means something in a side-by-side view.
export function Workbench({ desk, sheet }: { desk: React.ReactNode; sheet: React.ReactNode }) {
  const width = useResumeStore((s) => s.sidebarWidth);
  const collapsed = useResumeStore((s) => s.sidebarCollapsed);
  const setWidth = useResumeStore((s) => s.setSidebarWidth);
  const toggle = useResumeStore((s) => s.toggleSidebar);
  const [dragging, setDragging] = useState(false);
  const frame = useRef<HTMLDivElement>(null);

  const onMove = useCallback((clientX: number) => {
    const left = frame.current?.getBoundingClientRect().left ?? 0;
    setWidth(clientX - left);
  }, [setWidth]);

  useEffect(() => {
    if (!dragging) return;
    const move = (e: PointerEvent) => { e.preventDefault(); onMove(e.clientX); };
    const up = () => setDragging(false);
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [dragging, onMove]);

  // Keyboard resize on the handle — accessibility for a pointer-only control.
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowLeft") { setWidth(width - 24); e.preventDefault(); }
    if (e.key === "ArrowRight") { setWidth(width + 24); e.preventDefault(); }
  };

  return (
    <div
      ref={frame}
      className="flex min-h-screen flex-col lg:flex-row"
      style={{ ["--desk-w" as string]: collapsed ? "0px" : `${width}px` }}
    >
      <div
        data-desk
        className="no-print w-full shrink-0 overflow-hidden border-hairline transition-[width] duration-200 ease-out lg:w-[var(--desk-w)] lg:border-r"
      >
        {!collapsed && desk}
      </div>

      {/* Resize handle — hidden on mobile (stacked), hidden when collapsed */}
      {!collapsed && (
        <div
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize editor"
          tabIndex={0}
          onKeyDown={onKey}
          onPointerDown={(e) => { (e.target as HTMLElement).setPointerCapture?.(e.pointerId); setDragging(true); }}
          className={`no-print group relative hidden w-0 shrink-0 lg:block ${dragging ? "z-20" : ""}`}
        >
          <span
            className={`absolute -left-1.5 top-0 h-full w-3 cursor-col-resize`}
            aria-hidden
          />
          <span
            className={`pointer-events-none absolute left-0 top-1/2 h-10 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full transition-colors ${dragging ? "bg-studio" : "bg-hairline group-hover:bg-studio"}`}
            aria-hidden
          />
        </div>
      )}

      {/* Collapse / expand toggle, pinned to the seam */}
      <button
        onClick={toggle}
        aria-label={collapsed ? "Show editor" : "Hide editor"}
        className="no-print fixed bottom-5 left-5 z-30 hidden h-9 items-center gap-1.5 rounded-full border border-hairline bg-surface-raised px-3 font-label text-[10px] uppercase tracking-[0.14em] text-ink-muted shadow-sheet transition-colors hover:text-studio lg:flex"
      >
        <span aria-hidden className="text-studio">{collapsed ? "»" : "«"}</span>
        {collapsed ? "Editor" : "Hide"}
      </button>

      <div className="print-area min-w-0 flex-1 overflow-y-auto bg-desk lg:h-screen">{sheet}</div>
    </div>
  );
}
