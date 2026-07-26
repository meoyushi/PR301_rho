import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ResumePreview } from "./ResumePreview";

const style = { fontSize: 14, margin: 48, lineSpacing: 1.4, accent: "#000", sectionOrder: ["summary", "skills", "work", "education"] };
const resume = {
  name: "Jane Doe", headline: "Engineer", summary: "Builds things",
  emails: [], phones: [], urls: [], education: [], certifications: [],
  skills: ["python", "aws"],
  work: [{ company: "Acme", title: "Engineer", bullets: ["Built X in Python"] }],
};

describe("ResumePreview", () => {
  it("renders name, skills and bullets", () => {
    render(<ResumePreview resume={resume as any} style={style as any} optimize={null} />);
    expect(screen.getByText("Jane Doe")).toBeDefined();
    expect(screen.getByText("python")).toBeDefined();
    expect(screen.getByText(/Built X in Python/)).toBeDefined();
  });

  it("shows before/after when a bullet was tailored", () => {
    const original = { ...resume, work: [{ company: "Acme", title: "Engineer", bullets: ["Built X"] }] };
    render(<ResumePreview resume={resume as any} style={style as any}
      optimize={{ score: 80, previousScore: 60, gaps: [], fabricationsBlocked: 0, originalResume: original } as any} />);
    // original struck-through text present alongside the tailored version
    expect(screen.getByText("Built X")).toBeDefined();
    expect(screen.getByText(/Built X in Python/)).toBeDefined();
  });

  it("renders an achievements section", () => {
    const withAch = { ...resume, achievements: ["Winner, ICPC 2021", "Patent US123"] };
    const st = { ...style, sectionOrder: ["summary", "achievements", "skills"] };
    render(<ResumePreview resume={withAch as any} style={st as any} optimize={null} />);
    expect(screen.getByText("Achievements")).toBeDefined();
    expect(screen.getByText("Winner, ICPC 2021")).toBeDefined();
    expect(screen.getByText("Patent US123")).toBeDefined();
  });

  it("hides a section listed in hiddenSections", () => {
    const st = { ...style, sectionOrder: ["summary", "skills"], hiddenSections: ["skills"] };
    render(<ResumePreview resume={resume as any} style={st as any} optimize={null} />);
    expect(screen.queryByText("python")).toBeNull();
  });

  it("renders the modern template as two columns", () => {
    const st = { ...style, template: "modern", sectionOrder: ["summary", "skills", "work", "education"] };
    const { container } = render(<ResumePreview resume={resume as any} style={st as any} optimize={null} />);
    // two-column layout emits a CSS grid wrapper the single-column path does not
    expect(container.querySelector(".grid")).not.toBeNull();
    expect(screen.getByText("Jane Doe")).toBeDefined();
  });

  it("renders a projects section with name, tech and bullets", () => {
    const withProjects = {
      ...resume,
      projects: [{ name: "CredVault", url: "https://x", tech: ["Python", "Redis"], bullets: ["Built auth handling 2TB/day"] }],
    };
    const styleWithProjects = { ...style, sectionOrder: ["summary", "skills", "work", "projects", "education"] };
    render(<ResumePreview resume={withProjects as any} style={styleWithProjects as any} optimize={null} />);
    expect(screen.getByText("CredVault")).toBeDefined();
    expect(screen.getByText("Python, Redis")).toBeDefined();
    expect(screen.getByText(/Built auth handling 2TB\/day/)).toBeDefined();
  });
});
