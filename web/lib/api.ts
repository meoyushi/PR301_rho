import type { JobStatus, ParseResponse, StructuredResume } from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class BackendUnreachable extends Error {}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail ?? res.statusText);
  return res.json() as Promise<T>;
}

export async function parseResume(file: File): Promise<ParseResponse> {
  const form = new FormData();
  form.append("file", file);
  let res: Response;
  try { res = await fetch(`${BASE}/parse`, { method: "POST", body: form }); }
  catch { throw new BackendUnreachable("backend unreachable"); }
  return json<ParseResponse>(res);
}

export async function startOptimize(resume: StructuredResume, jdText: string): Promise<JobStatus> {
  let res: Response;
  try {
    res = await fetch(`${BASE}/optimize`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ resume, jd_text: jdText }),
    });
  } catch { throw new BackendUnreachable("backend unreachable"); }
  return json<JobStatus>(res);
}

export async function downloadDocx(
  resume: StructuredResume,
  sectionOrder: string[],
  accent: string,
  hiddenSections: string[] = [],
): Promise<Blob> {
  let res: Response;
  try {
    res = await fetch(`${BASE}/export/docx`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ resume, section_order: sectionOrder, accent, hidden_sections: hiddenSections }),
    });
  } catch { throw new BackendUnreachable("backend unreachable"); }
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail ?? res.statusText);
  return res.blob();
}

export async function pollOptimize(
  jobId: string,
  { intervalMs = 1000, timeoutMs = 120000 }: { intervalMs?: number; timeoutMs?: number } = {},
): Promise<JobStatus> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    let res: Response;
    try { res = await fetch(`${BASE}/optimize/${jobId}`); }
    catch { throw new BackendUnreachable("backend unreachable"); }
    const js = await json<JobStatus>(res);
    if (js.state === "done") return js;
    if (js.state === "error") throw new Error(js.error ?? "optimize failed");
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new Error("optimize timed out");
}
