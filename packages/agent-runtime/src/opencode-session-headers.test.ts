import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";
import {
  createAssistantMessageEventStream,
  type AssistantMessage,
  type Model,
  type SimpleStreamOptions,
} from "@earendil-works/pi-ai";
import { completeOneShot } from "./one-shot-complete.js";
import {
  OPENCODE_CLIENT_HEADER,
  OPENCODE_CLIENT_VALUE,
  OPENCODE_CLI_VERSION,
  OPENCODE_REQUEST_HEADER,
  OPENCODE_SESSION_HEADER,
  OPENCODE_USER_AGENT,
  deriveOpenCodeSessionId,
  isOpenCodeRequestId,
  isOpenCodeSessionId,
  isOpenCodeEndpoint,
  mergeOpenCodeSessionHeaders,
  newOpenCodeRequestId,
  newOpenCodeSessionId,
  withOpenCodeSessionHeaders,
} from "./opencode-session-headers.js";
import type { RuntimeProviderConfig } from "./provider-binding.js";

const openaiCompletionsModel = {
  id: "glm-5.3-flash",
  name: "GLM-5.3-Flash",
  api: "openai-completions",
  provider: "row-uuid",
  baseUrl: "https://opencode.ai/zen/go/v1",
  reasoning: true,
  input: ["text"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 128_000,
  maxTokens: 8_192,
} as Model<"openai-completions">;

const localModel = {
  ...openaiCompletionsModel,
  provider: "local",
  baseUrl: "http://127.0.0.1:11434/v1",
} as Model<"openai-completions">;

describe("isOpenCodeEndpoint", () => {
  it("matches the OpenCode Go apiStyle even when the row id is a UUID", () => {
    expect(
      isOpenCodeEndpoint({
        apiStyle: "opencode_go",
        baseUrl: "https://opencode.ai/zen/go/v1",
      }),
    ).toBe(true);
  });

  it("matches vendorKey and native pi-ai provider ids", () => {
    expect(isOpenCodeEndpoint({ vendorKey: "opencode-go" })).toBe(true);
    expect(isOpenCodeEndpoint({ vendorKey: "opencode" })).toBe(true);
    expect(
      isOpenCodeEndpoint({
        model: { ...openaiCompletionsModel, provider: "opencode-go" },
      }),
    ).toBe(true);
  });

  it("matches opencode.ai hosts on a generic OpenAI-compatible row", () => {
    expect(
      isOpenCodeEndpoint({
        apiStyle: "chat_completions",
        baseUrl: "https://opencode.ai/zen/go/v1",
      }),
    ).toBe(true);
    expect(
      isOpenCodeEndpoint({
        model: openaiCompletionsModel,
      }),
    ).toBe(true);
  });

  it("does not match other OpenAI-compatible gateways", () => {
    expect(
      isOpenCodeEndpoint({
        apiStyle: "chat_completions",
        baseUrl: "https://openrouter.ai/api/v1",
      }),
    ).toBe(false);
    expect(isOpenCodeEndpoint({ model: localModel })).toBe(false);
  });
});

describe("mergeOpenCodeSessionHeaders", () => {
  it("injects stamped request id with the native CLI identity", () => {
    const headers = mergeOpenCodeSessionHeaders({
      apiStyle: "opencode_go",
      sessionId: "session-1",
    });
    // merge is verbatim: the timestamp-encoded wire session id is derived by
    // withOpenCodeSessionHeaders (tested below). merge only adds what's missing.
    expect(headers?.[OPENCODE_SESSION_HEADER]).toBe("session-1");
    expect(headers?.[OPENCODE_REQUEST_HEADER]).toMatch(
      /^msg_[0-9a-f]{12}[A-Za-z0-9]{14}$/,
    );
    expect(headers).toMatchObject({
      [OPENCODE_CLIENT_HEADER]: "cli",
      "User-Agent": `opencode/${OPENCODE_CLI_VERSION}`,
    });
    expect(OPENCODE_CLIENT_VALUE).toBe("cli");
    expect(OPENCODE_USER_AGENT).toBe(`opencode/${OPENCODE_CLI_VERSION}`);
  });

  it("forces the native client identity but restores a missing session id", () => {
    expect(
      mergeOpenCodeSessionHeaders({
        apiStyle: "opencode_go",
        sessionId: "session-1",
        headers: {
          [OPENCODE_SESSION_HEADER]: null,
          [OPENCODE_CLIENT_HEADER]: "custom-client",
          "User-Agent": "third-party/9.9",
          "X-Extra": "keep",
        },
      }),
    ).toEqual({
      [OPENCODE_SESSION_HEADER]: "session-1",
      [OPENCODE_REQUEST_HEADER]: expect.stringMatching(
        /^msg_[0-9a-f]{12}[A-Za-z0-9]{14}$/,
      ),
      // A stale third-party client identity would fail the free-tier
      // signature, so it is forced — unlike ordinary caller headers.
      [OPENCODE_CLIENT_HEADER]: "cli",
      "User-Agent": OPENCODE_USER_AGENT,
      "X-Extra": "keep",
    });
  });

  it("keeps a genuine opencode User-Agent suffix", () => {
    expect(
      mergeOpenCodeSessionHeaders({
        apiStyle: "opencode_go",
        sessionId: "session-1",
        headers: { "User-Agent": "opencode/1.18.32 VercelAI/5.0" },
      })?.["User-Agent"],
    ).toBe("opencode/1.18.32 VercelAI/5.0");
  });

  it("preserves an explicit session header", () => {
    expect(
      mergeOpenCodeSessionHeaders({
        apiStyle: "opencode_go",
        sessionId: "session-1",
        headers: { [OPENCODE_SESSION_HEADER]: "already-set" },
      })?.[OPENCODE_SESSION_HEADER],
    ).toBe("already-set");
  });

  it("does not add headers for other providers", () => {
    expect(
      mergeOpenCodeSessionHeaders({
        apiStyle: "chat_completions",
        baseUrl: "https://api.openai.com/v1",
        sessionId: "session-1",
        headers: { "X-Existing": "1" },
      }),
    ).toEqual({ "X-Existing": "1" });
  });

  it("does not add headers without a session id", () => {
    expect(
      mergeOpenCodeSessionHeaders({
        apiStyle: "opencode_go",
        sessionId: "  ",
      }),
    ).toBeUndefined();
  });
});

describe("withOpenCodeSessionHeaders", () => {
  it("keeps the harness sessionId on the options and derives a stable wire id", () => {
    const options: SimpleStreamOptions = { temperature: 0 };
    const first = withOpenCodeSessionHeaders(options, {
      apiStyle: "opencode_go",
      sessionId: "session-1",
    });
    const second = withOpenCodeSessionHeaders(first, {
      apiStyle: "opencode_go",
    });
    // Routing still uses the harness id; the wire header carries the derived
    // timestamp-encoded id the gateway pins the conversation by.
    expect(first.sessionId).toBe("session-1");
    expect(second.sessionId).toBe("session-1");
    expect(first.headers?.[OPENCODE_SESSION_HEADER]).toBe(
      deriveOpenCodeSessionId("session-1"),
    );
    expect(second.headers?.[OPENCODE_SESSION_HEADER]).toBe(
      first.headers?.[OPENCODE_SESSION_HEADER],
    );
    // A retry reuses the same options object, so the request id it already
    // carries is preserved and the retry stays byte-identical.
    expect(first.headers?.[OPENCODE_REQUEST_HEADER]).toMatch(
      /^msg_[0-9a-f]{12}[A-Za-z0-9]{14}$/,
    );
    expect(second.headers?.[OPENCODE_REQUEST_HEADER]).toBe(
      first.headers?.[OPENCODE_REQUEST_HEADER],
    );
    expect(first.temperature).toBe(0);
  });

  it("synthesizes a stamped session id for OpenCode one-shot calls that have none", () => {
    const result = withOpenCodeSessionHeaders({}, { apiStyle: "opencode_go" });
    expect(result.sessionId).toMatch(/^ses_[0-9a-f]{12}[A-Za-z0-9]{14}$/);
    expect(result.headers?.[OPENCODE_SESSION_HEADER]).toBe(result.sessionId);
  });

  it("does not synthesize a session id for other providers", () => {
    const result = withOpenCodeSessionHeaders(
      { temperature: 1 },
      { apiStyle: "chat_completions", baseUrl: "https://api.openai.com/v1" },
    );
    expect(result.sessionId).toBeUndefined();
    expect(result.headers).toBeUndefined();
    expect(result.temperature).toBe(1);
  });
});

describe("completeOneShot OpenCode headers", () => {
  const provider: RuntimeProviderConfig = {
    id: "row-uuid",
    name: "OpenCode Go",
    vendorKey: "opencode-go",
    baseUrl: "https://opencode.ai/zen/go/v1",
    modelId: "glm-5.3-flash",
    apiKey: "sk-test",
    apiStyle: "opencode_go",
    supportsReasoning: false,
    supportedThinkingLevels: ["off"],
  };

  function assistantOk(): AssistantMessage {
    return {
      role: "assistant",
      content: [{ type: "text", text: "ok" }],
      api: "openai-completions",
      provider: provider.id,
      model: provider.modelId,
      usage: {
        input: 1,
        output: 1,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 2,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      stopReason: "stop",
      timestamp: Date.now(),
    };
  }

  function streamFor(message: AssistantMessage) {
    const stream = createAssistantMessageEventStream();
    queueMicrotask(() => {
      stream.push({ type: "start", partial: message });
      stream.push({ type: "done", reason: "stop", message });
      stream.end(message);
    });
    return stream;
  }

  it("sends the conversation id on OpenCode Go one-shot completions", async () => {
    let captured: SimpleStreamOptions | undefined;
    const result = await completeOneShot(
      provider,
      { systemPrompt: "s", messages: [] },
      "off",
      {
        sessionId: "session-9",
        stream: (_model, _context, options) => {
          captured = options;
          return streamFor(assistantOk());
        },
      },
    );
    expect(result.text).toBe("ok");
    expect(result.usage).toMatchObject({ operationId: expect.any(String), providerId: provider.id, modelId: provider.modelId });
    expect(captured?.sessionId).toBe("session-9");
    expect(captured?.headers?.[OPENCODE_SESSION_HEADER]).toBe(
      deriveOpenCodeSessionId("session-9"),
    );
    expect(captured?.headers).toMatchObject({
      [OPENCODE_CLIENT_HEADER]: "cli",
      "User-Agent": OPENCODE_USER_AGENT,
      [OPENCODE_REQUEST_HEADER]: expect.stringMatching(
        /^msg_[0-9a-f]{12}[A-Za-z0-9]{14}$/,
      ),
    });
  });

  it("honors a caller output-token ceiling without changing the default budget", async () => {
    const capturedBudgets: number[] = [];
    await completeOneShot(provider, { systemPrompt: "s", messages: [] }, "off", {
      maxOutputTokens: 256,
      stream: (_model, _context, options) => {
        if (options?.maxTokens !== undefined) capturedBudgets.push(options.maxTokens);
        return streamFor(assistantOk());
      },
    });
    expect(capturedBudgets[0]).toBe(256);

    await completeOneShot(provider, { systemPrompt: "s", messages: [] }, "off", {
      stream: (_model, _context, options) => {
        if (options?.maxTokens !== undefined) capturedBudgets.push(options.maxTokens);
        return streamFor(assistantOk());
      },
    });
    expect(capturedBudgets[1]).toBeGreaterThan(256);
  });

  it("forwards caller cancellation to the provider stream", async () => {
    const controller = new AbortController();
    let captured: SimpleStreamOptions | undefined;
    await completeOneShot(provider, { systemPrompt: "s", messages: [] }, "off", {
      signal: controller.signal,
      stream: (_model, _context, options) => {
        captured = options;
        return streamFor(assistantOk());
      },
    });
    expect(captured?.signal).toBe(controller.signal);
  });

  it("does not attach OpenCode headers to a generic Completions provider", async () => {
    let captured: SimpleStreamOptions | undefined;
    await completeOneShot(
      {
        ...provider,
        apiStyle: "chat_completions",
        vendorKey: "openai",
        baseUrl: "https://api.openai.com/v1",
      },
      { systemPrompt: "s", messages: [] },
      "off",
      {
        sessionId: "session-9",
        stream: (_model, _context, options) => {
          captured = options;
          return streamFor(assistantOk());
        },
      },
    );
    expect(captured?.sessionId).toBe("session-9");
    expect(captured?.headers?.[OPENCODE_SESSION_HEADER]).toBeUndefined();
  });

  it("keeps a certificate rejection terminal in one-shot completions", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockRejectedValue(
      Object.assign(new TypeError("fetch failed"), {
        cause: Object.assign(new Error("certificate rejected"), {
          code: "SELF_SIGNED_CERT_IN_CHAIN",
        }),
      }),
    );
    let attempts = 0;
    try {
      await expect(
        completeOneShot(
          provider,
          { systemPrompt: "s", messages: [] },
          "off",
          {
            stream: (_model, _context, options) => {
              attempts += 1;
              const stream = createAssistantMessageEventStream();
              const failed = {
                ...assistantOk(),
                content: [],
                stopReason: "error" as const,
                errorMessage: "fetch failed",
              };
              void options?.fetch?.("https://provider.invalid", {}).then(
                () => {
                  stream.push({ type: "error", reason: "error", error: failed });
                  stream.end(failed);
                },
                () => {
                  stream.push({ type: "error", reason: "error", error: failed });
                  stream.end(failed);
                },
              );
              return stream;
            },
          },
        ),
      ).rejects.toMatchObject({
        errorCode: "NETWORK_ERROR",
        data: {
          networkCode: "SELF_SIGNED_CERT_IN_CHAIN",
          retriable: false,
        },
      });
      expect(attempts).toBe(1);
    } finally {
      fetchSpy.mockRestore();
    }
  });
});

describe("stamped OpenCode ids", () => {
  it("mints ses_/msg_ ids in the native shape", () => {
    expect(newOpenCodeSessionId()).toMatch(/^ses_[0-9a-f]{12}[A-Za-z0-9]{14}$/);
    expect(newOpenCodeRequestId()).toMatch(/^msg_[0-9a-f]{12}[A-Za-z0-9]{14}$/);
    expect(isOpenCodeSessionId(newOpenCodeSessionId())).toBe(true);
    expect(isOpenCodeRequestId(newOpenCodeRequestId())).toBe(true);
  });

  it("rejects UUID, random, and cross-prefix ids", () => {
    expect(isOpenCodeSessionId("123e4567-e89b-12d3-a456-426614174000")).toBe(false);
    expect(isOpenCodeSessionId("ses_" + "x".repeat(26))).toBe(false);
    expect(isOpenCodeSessionId(newOpenCodeRequestId())).toBe(false);
    expect(isOpenCodeRequestId(newOpenCodeSessionId())).toBe(false);
    expect(isOpenCodeSessionId("  ")).toBe(false);
  });

  it("derives one stable id per harness conversation", () => {
    const first = deriveOpenCodeSessionId("conv-7");
    expect(first).toMatch(/^ses_[0-9a-f]{12}[A-Za-z0-9]{14}$/);
    expect(deriveOpenCodeSessionId("conv-7")).toBe(first);
    expect(deriveOpenCodeSessionId("conv-8")).not.toBe(first);
  });

  it("mints a fresh id for ad-hoc calls without a conversation", () => {
    expect(deriveOpenCodeSessionId("   ")).toMatch(
      /^ses_[0-9a-f]{12}[A-Za-z0-9]{14}$/,
    );
    expect(deriveOpenCodeSessionId("")).not.toBe(deriveOpenCodeSessionId(""));
  });
});

describe("OpenCode header call-site wiring", () => {
  it("is applied on session, subagent, and one-shot streams", () => {
    const sources = [
      readFileSync(new URL("./runtime.ts", import.meta.url), "utf8"),
      readFileSync(new URL("./subagent-model-binding.ts", import.meta.url), "utf8"),
      readFileSync(new URL("./one-shot-complete.ts", import.meta.url), "utf8"),
    ];
    for (const source of sources) {
      expect(source).toContain("withOpenCodeSessionHeaders");
      expect(source).toContain("openCodeEndpointFromProvider");
    }
  });
});
