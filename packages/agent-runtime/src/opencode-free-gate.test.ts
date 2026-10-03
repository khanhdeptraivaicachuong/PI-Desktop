import { describe, expect, it, vi } from "vitest";
import {
  createOpenCodeFreeGateFetch,
  gateOpenCodeFreeBody,
  isOpenCodeFreeTierModelId,
  isOpenCodeZenRequest,
  withOpenCodeFreeGate,
} from "./opencode-free-gate.js";

const CHAT_BODY = {
  model: "mimo-v2.5-free",
  messages: [{ role: "user", content: "hi" }],
  stream: true,
};

describe("isOpenCodeFreeTierModelId", () => {
  it("matches -free ids case-insensitively", () => {
    expect(isOpenCodeFreeTierModelId("mimo-v2.5-free")).toBe(true);
    expect(isOpenCodeFreeTierModelId("  Muse-Spark-1.3-Contributor-Free ")).toBe(true);
    expect(isOpenCodeFreeTierModelId("gpt-5")).toBe(false);
    expect(isOpenCodeFreeTierModelId(undefined)).toBe(false);
  });
});

describe("isOpenCodeZenRequest", () => {
  it("matches the zen host and subdomains over https only", () => {
    expect(isOpenCodeZenRequest("https://opencode.ai/zen/v1/chat/completions")).toBe(true);
    expect(isOpenCodeZenRequest("https://us.opencode.ai/zen/v1/models")).toBe(true);
    expect(isOpenCodeZenRequest("http://opencode.ai/zen/v1/chat/completions")).toBe(false);
    expect(isOpenCodeZenRequest("https://openrouter.ai/api/v1/chat/completions")).toBe(false);
    expect(isOpenCodeZenRequest("not a url")).toBe(false);
  });
});

describe("withOpenCodeFreeGate", () => {
  it("merges marker tools and a system prompt into chat bodies", () => {
    const { payload, changed } = withOpenCodeFreeGate({ ...CHAT_BODY });
    expect(changed).toBe(true);
    const body = payload as Record<string, unknown>;
    expect(body.stream).toBe(true);
    const messages = body.messages as Array<Record<string, unknown>>;
    expect(messages[0]).toEqual({ role: "system", content: expect.any(String) });
    const names = (body.tools as Array<Record<string, unknown>>).map(
      (tool) => (tool.function as Record<string, unknown>).name,
    );
    expect(names).toEqual(expect.arrayContaining(["read", "write", "edit", "bash"]));
  });

  it("keeps the caller's real tools and never flips stream", () => {
    const realTool = {
      type: "function",
      function: {
        name: "Read",
        description: "read a file",
        parameters: { type: "object", properties: {} },
      },
    };
    const { payload } = withOpenCodeFreeGate({
      model: "mimo-v2.5-free",
      messages: [{ role: "system", content: "s" }, { role: "user", content: "hi" }],
      tools: [realTool],
      stream: false,
    });
    const body = payload as Record<string, unknown>;
    expect(body.stream).toBe(false);
    expect(body.tools).toEqual(
      expect.arrayContaining([realTool, expect.objectContaining({ type: "function" })]),
    );
    // Existing system prompt is kept, not duplicated.
    expect((body.messages as unknown[]).filter(
      (message) => (message as Record<string, unknown>).role === "system",
    )).toHaveLength(1);
  });

  it("gates responses bodies with instructions and the responses envelope", () => {
    const { payload, changed } = withOpenCodeFreeGate({
      model: "muse-spark-1.3-contributor-free",
      instructions: "  ",
      input: [{ role: "user", content: "hi" }],
      stream: true,
    });
    expect(changed).toBe(true);
    const body = payload as Record<string, unknown>;
    expect(body.instructions).toBe("You are a helpful coding assistant.");
    expect(body.tools).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: "function", name: "read" }),
        expect.objectContaining({ type: "function", name: "bash" }),
      ]),
    );
  });

  it("passes malformed or foreign payloads through untouched", () => {
    expect(withOpenCodeFreeGate({ model: "x", tools: "nope" }).changed).toBe(false);
    expect(withOpenCodeFreeGate("text").changed).toBe(false);
    expect(withOpenCodeFreeGate({ model: "x" }).changed).toBe(false);
  });
});

describe("gateOpenCodeFreeBody", () => {
  it("reports gated, unchanged, and incomplete", () => {
    expect(gateOpenCodeFreeBody(JSON.stringify(CHAT_BODY)).status).toBe("gated");
    const gated = gateOpenCodeFreeBody(JSON.stringify(CHAT_BODY));
    expect(gated.status).toBe("gated");
    // Gating twice is stable: the second pass finds everything present.
    if (gated.status === "gated") {
      expect(gateOpenCodeFreeBody(gated.body).status).toBe("unchanged");
    }
    expect(gateOpenCodeFreeBody("not json").status).toBe("incomplete");
    expect(gateOpenCodeFreeBody(undefined).status).toBe("incomplete");
    expect(gateOpenCodeFreeBody(JSON.stringify({ model: "x" })).status).toBe("incomplete");
  });
});

describe("createOpenCodeFreeGateFetch", () => {
  function mockBase() {
    const calls: Array<{ input: unknown; init: unknown }> = [];
    const base = vi.fn(async (input: unknown, init: unknown) => {
      calls.push({ input, init });
      return new Response("{}");
    });
    return { calls, base };
  }

  it("gates zen free-tier bodies and passes the rest through", async () => {
    const { calls, base } = mockBase();
    const fetch = createOpenCodeFreeGateFetch(base as never);
    await fetch("https://opencode.ai/zen/v1/chat/completions", {
      method: "POST",
      body: JSON.stringify(CHAT_BODY),
    });
    expect(base).toHaveBeenCalledTimes(1);
    const sent = JSON.parse(String((calls[0]?.init as Record<string, unknown>).body));
    expect(sent.tools).toEqual(
      expect.arrayContaining([expect.objectContaining({ type: "function" })]),
    );
    expect(sent.messages[0].role).toBe("system");
  });

  it("leaves paid models and foreign hosts byte-identical", async () => {
    const { calls, base } = mockBase();
    const fetch = createOpenCodeFreeGateFetch(base as never);
    const paidBody = JSON.stringify({ ...CHAT_BODY, model: "gpt-5" });
    await fetch("https://opencode.ai/zen/v1/chat/completions", {
      method: "POST",
      body: paidBody,
    });
    await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      body: JSON.stringify(CHAT_BODY),
    });
    expect(calls[0]?.init).toMatchObject({ body: paidBody });
    expect(calls[1]?.input).toBe("https://openrouter.ai/api/v1/chat/completions");
  });
});
