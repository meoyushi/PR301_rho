"use client";
import { UploadDropzone } from "@/components/UploadDropzone";
import { FieldEditors } from "@/components/Editor/FieldEditors";
import { WorkEditor } from "@/components/Editor/WorkEditor";
import { ProjectsEditor } from "@/components/Editor/ProjectsEditor";
import { AchievementsEditor } from "@/components/Editor/AchievementsEditor";
import { SkillsEditor } from "@/components/Editor/SkillsEditor";
import { StyleControls } from "@/components/Editor/StyleControls";
import { TemplatePicker } from "@/components/Editor/TemplatePicker";
import { SectionArranger } from "@/components/Editor/SectionArranger";
import { JdBox } from "@/components/Editor/JdBox";
import { DownloadBar } from "@/components/Editor/DownloadBar";
import { ResumePreview } from "@/components/Preview/ResumePreview";
import { ThemeToggle } from "@/components/ThemeToggle";
import { Workbench } from "@/components/Workbench";
import { useResumeStore } from "@/lib/resumeStore";

// A grouped panel on the desk: mono eyebrow label + content card.
function Panel({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <section className="space-y-2.5">
      <h2 className="font-label text-[11px] uppercase tracking-[0.18em] text-ink-muted">{label}</h2>
      {children}
    </section>
  );
}

export default function Page() {
  const resume = useResumeStore((s) => s.resume);
  const style = useResumeStore((s) => s.style);
  const optimize = useResumeStore((s) => s.optimize);

  const desk = (
    <div className="flex flex-col lg:h-screen">
      {/* Sticky desk header: wordmark + theme */}
      <header className="flex items-center justify-between border-b border-hairline bg-desk/80 px-6 py-4 backdrop-blur">
        <div className="flex items-baseline gap-2">
          <span className="font-label text-[13px] font-semibold uppercase tracking-[0.3em] text-studio">rho</span>
          <span className="font-label text-[10px] uppercase tracking-[0.16em] text-ink-muted">résumé studio</span>
        </div>
        <ThemeToggle />
      </header>

      <div className="min-h-0 flex-1 space-y-7 overflow-y-auto px-6 py-6">
        <UploadDropzone />
        {resume && (
          <>
            {/* These editors carry their own section headings */}
            <FieldEditors />
            <SkillsEditor />
            <WorkEditor />
            <ProjectsEditor />
            <AchievementsEditor />

            <div className="border-t border-hairline pt-5"><Panel label="Template"><TemplatePicker /></Panel></div>
            <Panel label="Sections">
              <p className="-mt-1 text-xs text-ink-muted">Drag to reorder. Toggle the eye to hide a section without deleting it.</p>
              <SectionArranger />
            </Panel>

            <details className="group border-t border-hairline pt-3">
              <summary className="cursor-pointer list-none font-label text-[11px] uppercase tracking-[0.18em] text-ink-muted transition-colors hover:text-studio">
                Fine-tune <span className="ml-1 inline-block transition-transform group-open:rotate-90">›</span>
              </summary>
              <StyleControls />
            </details>

            <div className="border-t border-hairline pt-1"><JdBox /></div>
            <DownloadBar />
          </>
        )}
      </div>
    </div>
  );

  const sheet = (
    <div className="flex min-h-full justify-center px-6 py-10">
      {resume ? (
        <div className="w-full max-w-3xl">
          <ResumePreview resume={resume} style={style} optimize={optimize} />
        </div>
      ) : (
        <div className="flex min-h-[60vh] max-w-sm flex-col items-center justify-center text-center">
          <div className="mb-5 h-24 w-[4.5rem] rounded-sm border border-hairline bg-paper shadow-sheet" aria-hidden />
          <p className="font-label text-[11px] uppercase tracking-[0.18em] text-ink-muted">The sheet</p>
          <p className="mt-2 text-sm text-ink-muted">
            Upload a résumé on the left. It renders here as a formatted sheet you can template, arrange, and export.
          </p>
        </div>
      )}
    </div>
  );

  return <Workbench desk={desk} sheet={sheet} />;
}
