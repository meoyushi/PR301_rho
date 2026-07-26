import { beforeEach, describe, expect, it } from "vitest";
import { useResumeStore } from "./resumeStore";

const base = () => ({
  name: "Jane", headline: null, summary: null, emails: [], phones: [], urls: [],
  work: [{ company: "Acme", title: "Eng", bullets: ["Built X"] }],
  education: [],
  projects: [{ name: "CredVault", url: null, tech: ["Python"], bullets: ["Built auth"] }],
  skills: ["python"], certifications: [], achievements: [],
});

beforeEach(() => useResumeStore.getState().setResume(base() as any));

describe("resume store", () => {
  it("edits a top-level field", () => {
    useResumeStore.getState().setField("summary", "Senior engineer");
    expect(useResumeStore.getState().resume!.summary).toBe("Senior engineer");
  });

  it("adds and removes a bullet on a work entry", () => {
    useResumeStore.getState().addBullet(0);
    expect(useResumeStore.getState().resume!.work[0].bullets).toHaveLength(2);
    useResumeStore.getState().editBullet(0, 1, "Led Y");
    expect(useResumeStore.getState().resume!.work[0].bullets[1]).toBe("Led Y");
    useResumeStore.getState().removeBullet(0, 0);
    expect(useResumeStore.getState().resume!.work[0].bullets).toEqual(["Led Y"]);
  });

  it("adds, edits and removes a bullet on a project entry", () => {
    useResumeStore.getState().addProjectBullet(0);
    expect(useResumeStore.getState().resume!.projects[0].bullets).toHaveLength(2);
    useResumeStore.getState().editProjectBullet(0, 1, "Added caching");
    expect(useResumeStore.getState().resume!.projects[0].bullets[1]).toBe("Added caching");
    useResumeStore.getState().removeProjectBullet(0, 0);
    expect(useResumeStore.getState().resume!.projects[0].bullets).toEqual(["Added caching"]);
  });

  it("adds and removes skills without duplicates", () => {
    useResumeStore.getState().addSkill("python"); // dup ignored
    useResumeStore.getState().addSkill("aws");
    expect(useResumeStore.getState().resume!.skills).toEqual(["python", "aws"]);
    useResumeStore.getState().removeSkill("python");
    expect(useResumeStore.getState().resume!.skills).toEqual(["aws"]);
  });

  it("adds, edits and removes achievements", () => {
    useResumeStore.getState().addAchievement();
    useResumeStore.getState().editAchievement(0, "Winner, ICPC 2021");
    expect(useResumeStore.getState().resume!.achievements).toEqual(["Winner, ICPC 2021"]);
    useResumeStore.getState().removeAchievement(0);
    expect(useResumeStore.getState().resume!.achievements).toEqual([]);
  });

  it("updates style settings", () => {
    useResumeStore.getState().setStyle({ fontSize: 12 });
    expect(useResumeStore.getState().style.fontSize).toBe(12);
  });

  it("setTemplate swaps template and applies its default accent", () => {
    useResumeStore.getState().setTemplate("modern");
    const s = useResumeStore.getState().style;
    expect(s.template).toBe("modern");
    expect(s.accent).toBe("#2f5d8a");
  });

  it("moveSection reorders and ignores out-of-range indices", () => {
    useResumeStore.getState().setStyle({ sectionOrder: ["summary", "skills", "work", "projects", "education"] });
    useResumeStore.getState().moveSection(0, 2);
    expect(useResumeStore.getState().style.sectionOrder).toEqual(["skills", "work", "summary", "projects", "education"]);
    const before = useResumeStore.getState().style.sectionOrder;
    useResumeStore.getState().moveSection(0, 99); // out of range: no-op
    expect(useResumeStore.getState().style.sectionOrder).toEqual(before);
  });

  it("toggleSection hides then shows a section", () => {
    useResumeStore.getState().setStyle({ hiddenSections: [] });
    useResumeStore.getState().toggleSection("projects");
    expect(useResumeStore.getState().style.hiddenSections).toContain("projects");
    useResumeStore.getState().toggleSection("projects");
    expect(useResumeStore.getState().style.hiddenSections).not.toContain("projects");
  });

  it("toggleTheme flips light/dark", () => {
    const start = useResumeStore.getState().theme;
    useResumeStore.getState().toggleTheme();
    expect(useResumeStore.getState().theme).not.toBe(start);
  });

  it("setSidebarWidth clamps to the allowed range", () => {
    useResumeStore.getState().setSidebarWidth(50);   // below min
    expect(useResumeStore.getState().sidebarWidth).toBe(320);
    useResumeStore.getState().setSidebarWidth(9999); // above max
    expect(useResumeStore.getState().sidebarWidth).toBe(620);
  });

  it("applyOptimize swaps in tailored resume, sets scores and components", () => {
    useResumeStore.getState().applyOptimize({
      tailored: { name: "Jane", work: [{ company: "Acme", title: "Eng", bullets: ["Built X in Python"] }], skills: ["python"], emails: [], phones: [], urls: [], education: [], projects: [], certifications: [] } as any,
      displayScore: 88,
      baselineDisplayScore: 60,
      components: [{ label: "Keyword match", before: 0.5, after: 0.9 }],
      gaps: [],
      fabricationsBlocked: 2,
      previousScore: 55,
    });
    const s = useResumeStore.getState();
    expect(s.resume!.work[0].bullets[0]).toBe("Built X in Python");
    expect(s.optimize?.score).toBe(88);
    expect(s.optimize?.baselineScore).toBe(60);
    expect(s.optimize?.previousScore).toBe(55);
    expect(s.optimize?.components[0].after).toBe(0.9);
    expect(s.optimize?.fabricationsBlocked).toBe(2);
  });
});
