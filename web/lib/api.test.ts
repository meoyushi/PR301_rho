// web/lib/api.test.ts
import { afterEach, describe, expect, it, vi } from "vitest";
import { BackendUnreachable, pollOptimize, startOptimize } from "./api";

afterEach(() => vi.restoreAllMocks());

describe("api client", () => {
  it("startOptimize posts resume + jd and returns the job id", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ id: "j1", state: "queued" }) });
    vi.stubGlobal("fetch", fetchMock);
    const res = await startOptimize({ name: "X" } as any, "jd");
    expect(res.id).toBe("j1");
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it("pollOptimize resolves when the job reaches done", async () => {
    const states = [
      { id: "j1", state: "running", stage: "matching" },
      { id: "j1", state: "done", result: { final_score: 77 } },
    ];
    const fetchMock = vi.fn().mockImplementation(async () => ({ ok: true, json: async () => states.shift() }));
    vi.stubGlobal("fetch", fetchMock);
    const js = await pollOptimize("j1", { intervalMs: 1, timeoutMs: 1000 });
    expect(js.state).toBe("done");
    expect(js.result!.final_score).toBe(77);
  });

  it("pollOptimize rejects on the error state", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ id: "j1", state: "error", error: "model down" }) });
    vi.stubGlobal("fetch", fetchMock);
    await expect(pollOptimize("j1", { intervalMs: 1, timeoutMs: 1000 })).rejects.toThrow("model down");
  });

  it("pollOptimize throws a timeout error when the job never leaves running", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ id: "j1", state: "running" }) });
    vi.stubGlobal("fetch", fetchMock);
    await expect(pollOptimize("j1", { intervalMs: 5, timeoutMs: 20 })).rejects.toThrow("optimize timed out");
  });

  it("pollOptimize throws BackendUnreachable when fetch rejects", async () => {
    const fetchMock = vi.fn().mockRejectedValue(new TypeError("network error"));
    vi.stubGlobal("fetch", fetchMock);
    await expect(pollOptimize("j1", { intervalMs: 1, timeoutMs: 1000 })).rejects.toThrow(BackendUnreachable);
  });
});
