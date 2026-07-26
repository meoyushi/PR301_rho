"use client";
import type { StructuredResume } from "@/lib/types";
import type { OptimizeView, StyleSettings } from "@/lib/resumeStore";
import { TEMPLATES, type HeadingStyle, type Template } from "@/lib/templates";

function bulletBefore(original: StructuredResume | undefined, wi: number, bi: number): string | null {
  const b = original?.work?.[wi]?.bullets?.[bi];
  return b !== undefined ? b : null;
}
function projectBulletBefore(original: StructuredResume | undefined, pi: number, bi: number): string | null {
  const b = original?.projects?.[pi]?.bullets?.[bi];
  return b !== undefined ? b : null;
}

// Heading treatment is the most visible template signal. Each style is a
// className applied to the <h2>; the accent comes through CSS var --accent.
function headingClass(style: HeadingStyle): string {
  const base = "mt-4 mb-1 font-semibold text-[color:var(--accent)]";
  switch (style) {
    case "underline": return `${base} text-sm uppercase tracking-wide border-b border-[color:var(--accent)] pb-0.5`;
    case "bar": return `${base} text-sm uppercase tracking-wide pl-2 border-l-[3px] border-[color:var(--accent)]`;
    case "smallcaps": return `${base} text-xs uppercase tracking-[0.18em]`;
    case "block": return `${base} inline-block text-xs uppercase tracking-[0.14em] text-white px-2 py-0.5 rounded-sm`;
  }
}

export function ResumePreview({ resume, style, optimize }: {
  resume: StructuredResume; style: StyleSettings; optimize: OptimizeView | null;
}) {
  const template: Template = TEMPLATES[style.template] ?? TEMPLATES.classic;
  const hidden = new Set(style.hiddenSections ?? []);
  const sheet: React.CSSProperties = {
    fontSize: style.fontSize,
    padding: style.margin,
    lineHeight: style.lineSpacing,
    fontFamily: template.bodyFont,
    ["--accent" as string]: style.accent,
  };

  const H = ({ children }: { children: string }) => {
    const cls = headingClass(template.heading);
    const blockBg = template.heading === "block" ? { background: "var(--accent)" } : undefined;
    return <h2 className={cls} style={blockBg}>{children}</h2>;
  };

  const Summary = () =>
    !hidden.has("summary") && resume.summary
      ? <section key="summary"><H>Summary</H><p>{resume.summary}</p></section> : null;

  const Skills = () =>
    !hidden.has("skills") && resume.skills.length
      ? <section key="skills"><H>Skills</H>
          <ul className="flex flex-wrap gap-1.5">
            {resume.skills.map((s) => (
              <li key={s} className="rounded px-2 py-0.5 text-[0.85em]"
                  style={{ background: "color-mix(in srgb, var(--accent) 10%, transparent)" }}>{s}</li>
            ))}
          </ul>
        </section> : null;

  const Work = () =>
    !hidden.has("work") && resume.work.length
      ? <section key="work"><H>Experience</H>
          {resume.work.map((w, wi) => (
            <div key={wi} className="mt-2">
              <div className="flex justify-between gap-2"><strong>{w.title}</strong>
                <span className="text-[color:var(--sheet-ink-muted)]">{w.start_date}{w.end_date ? `–${w.end_date}` : ""}</span>
              </div>
              <div className="italic text-[color:var(--sheet-ink-muted)]">{w.company}</div>
              <ul className="mt-0.5 list-disc pl-5">
                {w.bullets.map((b, bi) => {
                  const before = optimize ? bulletBefore(optimize.originalResume, wi, bi) : null;
                  const changed = before !== null && before !== b;
                  return (
                    <li key={bi}>
                      {changed && <span className="print-hide mr-1 text-neutral-400 line-through">{before}</span>}
                      <span>{b}</span>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </section> : null;

  const Projects = () =>
    !hidden.has("projects") && resume.projects?.length
      ? <section key="projects"><H>Projects</H>
          {resume.projects.map((p, pi) => (
            <div key={pi} className="mt-2">
              <div className="flex justify-between gap-2"><strong>{p.name}</strong>
                {p.url && <a href={p.url} className="text-[color:var(--accent)] underline" target="_blank" rel="noreferrer">link</a>}
              </div>
              {p.tech.length > 0 && <div className="text-[0.9em] italic text-[color:var(--sheet-ink-muted)]">{p.tech.join(", ")}</div>}
              <ul className="mt-0.5 list-disc pl-5">
                {p.bullets.map((b, bi) => {
                  const before = optimize ? projectBulletBefore(optimize.originalResume, pi, bi) : null;
                  const changed = before !== null && before !== b;
                  return (
                    <li key={bi}>
                      {changed && <span className="print-hide mr-1 text-neutral-400 line-through">{before}</span>}
                      <span>{b}</span>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </section> : null;

  const Achievements = () =>
    !hidden.has("achievements") && resume.achievements?.length
      ? <section key="achievements"><H>Achievements</H>
          <ul className="mt-0.5 list-disc pl-5">
            {resume.achievements.map((a, ai) => <li key={ai}>{a}</li>)}
          </ul>
        </section> : null;

  const Education = () =>
    !hidden.has("education") && resume.education.length
      ? <section key="education"><H>Education</H>
          {resume.education.map((e, ei) => (
            <div key={ei} className="mt-1">{e.institution}{e.degree ? ` — ${e.degree}` : ""}{e.field ? `, ${e.field}` : ""}{e.end_year ? ` (${e.end_year})` : ""}</div>
          ))}
        </section> : null;

  const RENDERERS: Record<string, () => React.ReactNode> = {
    summary: Summary, skills: Skills, work: Work, projects: Projects,
    achievements: Achievements, education: Education,
  };
  const ordered = style.sectionOrder.map((k) => <div key={k}>{RENDERERS[k]?.()}</div>);

  const contact = [
    ...(resume.emails ?? []), ...(resume.phones ?? []), ...(resume.urls ?? []),
  ].filter(Boolean);

  const Header = (
    <header className={template.nameAlign === "center" ? "text-center" : ""}>
      <h1 className={`font-bold ${template.heading === "block" ? "text-3xl" : "text-2xl"}`}
          style={{ fontFamily: template.displayFont }}>{resume.name}</h1>
      {resume.headline && <p className="text-[color:var(--accent)]">{resume.headline}</p>}
      {contact.length > 0 && (
        <p className="mt-0.5 text-[0.85em] text-[color:var(--sheet-ink-muted)]">{contact.join("  ·  ")}</p>
      )}
    </header>
  );

  // Two-column layout (Modern): a left rail carries skills/education/contact,
  // the main column carries the narrative sections. Sections not in a column's
  // set fall through in document order.
  if (template.layout === "two-column") {
    const railKeys = ["skills", "education"];
    const rail = style.sectionOrder.filter((k) => railKeys.includes(k));
    const main = style.sectionOrder.filter((k) => !railKeys.includes(k));
    return (
      <article style={sheet} className="print-sheet mx-auto max-w-3xl bg-paper text-[color:var(--sheet-ink)] shadow-sheet">
        {Header}
        <div className="mt-3 grid grid-cols-[minmax(0,1fr)_minmax(0,2.1fr)] gap-6">
          <div>{rail.map((k) => <div key={k}>{RENDERERS[k]?.()}</div>)}</div>
          <div>{main.map((k) => <div key={k}>{RENDERERS[k]?.()}</div>)}</div>
        </div>
      </article>
    );
  }

  return (
    <article style={sheet} className="print-sheet mx-auto max-w-3xl bg-paper text-[color:var(--sheet-ink)] shadow-sheet">
      {Header}
      {ordered}
    </article>
  );
}
