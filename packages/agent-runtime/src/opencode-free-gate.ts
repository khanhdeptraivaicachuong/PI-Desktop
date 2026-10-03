/**
 * Free-tier body gate for the OpenCode Zen lane (`*-free` models).
 *
 * The gateway only answers requests that look like the native opencode CLI:
 * besides the `ses_`/`msg_` ids and `cli` identity (see
 * `./opencode-session-headers.js`), agent turns must carry lowercase marker
 * tools and a non-empty system prompt. pi-ai builds request bodies internally,
 * so the gate runs as a `fetch` wrapper: it parses the outgoing JSON, merges
 * the missing pieces in the protocol-correct envelope, and passes everything
 * else through byte-identical. Paid models and non-OpenCode hosts are never
 * touched.
 */

import type { FetchFunction } from "@earendil-works/pi-ai";

const OPENCODE_ROOT_DOMAIN = "opencode.ai";
const FREE_MODEL_SUFFIX = "-free";
const DEFAULT_SYSTEM_PROMPT = "You are a helpful coding assistant.";

// Marker tools exist only for lane recognition; the model must not call them.
// They coexist with the caller's real tools (case-mixed passes the gateway).
const MARKER_TOOL_NAMES = ["read", "write", "edit", "bash"] as const;
const MARKER_TOOL_DESCRIPTIONS: Readonly<Record<string, string>> = {
  read: "Do not call this tool. It only marks the request as coming from a coding agent.",
  write: "Do not call this tool. It only marks the request as coming from a coding agent.",
  edit: "Do not call this tool. It only marks the request as coming from a coding agent.",
  bash: "Do not call this tool. It only marks the request as coming from a coding agent.",
};

// The `model` field always sits at the top of the payload, so an 8KB probe
// keeps large bodies (images, long context) out of memory.
const MODEL_PROBE_MAX_CHARS = 8192;
const MODEL_FIELD_PATTERN = /"model"\s*:\s*"([^"\\]{1,200})/;
// Bodies past the bound (or not JSON objects) cannot be safely gated; the
// caller passes them through rather than half-signing a request.
const GATE_BODY_MAX_CHARS = 8 * 1024 * 1024;

export function isOpenCodeFreeTierModelId(modelId: unknown): boolean {
  if (typeof modelId !== "string") return false;
  return modelId.trim().toLowerCase().endsWith(FREE_MODEL_SUFFIX);
}

export function isOpenCodeZenRequest(requestUrl: string): boolean {
  let parsed: URL;
  try {
    parsed = new URL(requestUrl);
  } catch {
    return false;
  }
  if (parsed.protocol !== "https:") return false;
  const hostname = parsed.hostname.toLowerCase();
  return (
    hostname === OPENCODE_ROOT_DOMAIN ||
    hostname.endsWith(`.${OPENCODE_ROOT_DOMAIN}`)
  );
}

function toolNames(tools: unknown[]): Set<string> {
  const names = new Set<string>();
  for (const tool of tools) {
    if (!tool || typeof tool !== "object") continue;
    const direct = (tool as { name?: unknown }).name;
    if (typeof direct === "string") names.add(direct);
    const nested = (tool as { function?: unknown }).function;
    if (nested && typeof nested === "object") {
      const nestedName = (nested as { name?: unknown }).name;
      if (typeof nestedName === "string") names.add(nestedName);
    }
  }
  return names;
}

function hasMarkerTools(payload: Record<string, unknown>): boolean {
  if (!Array.isArray(payload.tools)) return false;
  const names = toolNames(payload.tools as unknown[]);
  return MARKER_TOOL_NAMES.every((name) => names.has(name));
}

function markerTool(chat: boolean, name: string): Record<string, unknown> {
  const description = MARKER_TOOL_DESCRIPTIONS[name];
  const parameters = { type: "object", properties: {} };
  return chat
    ? { type: "function", function: { name, description, parameters } }
    : { type: "function", name, description, parameters };
}

function nonEmptyText(value: unknown): string | undefined {
  if (typeof value !== "string" || !value.trim()) return undefined;
  return value;
}

/**
 * Merge the missing lowercase marker tools plus a non-empty system prompt
 * into a free-tier payload. Malformed `tools` are left for the gateway to
 * reject; `stream` and every other field pass through untouched.
 */
export function withOpenCodeFreeGate<T>(payload: T): { payload: T; changed: boolean } {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return { payload, changed: false };
  }
  const record = payload as Record<string, unknown>;
  if (record.tools !== undefined && !Array.isArray(record.tools)) {
    return { payload, changed: false };
  }
  const chat = Array.isArray(record.messages);
  const responses = !chat && (typeof record.instructions === "string" || Array.isArray(record.input));
  if (!chat && !responses) return { payload, changed: false };

  let changed = false;
  const next: Record<string, unknown> = { ...(record as Record<string, unknown>) };

  const tools = Array.isArray(record.tools) ? [...(record.tools as unknown[])] : [];
  const present = toolNames(tools);
  for (const name of MARKER_TOOL_NAMES) {
    if (present.has(name)) continue;
    tools.push(markerTool(chat, name));
    changed = true;
  }
  if (changed || !Array.isArray(record.tools)) {
    next.tools = tools;
    changed = true;
  }

  if (chat) {
    const messages = next.messages as Array<Record<string, unknown>>;
    const hasSystem = messages.some(
      (message) =>
        message?.role === "system" && nonEmptyText(message.content) !== undefined,
    );
    if (!hasSystem) {
      next.messages = [{ role: "system", content: DEFAULT_SYSTEM_PROMPT }, ...messages];
      changed = true;
    }
  } else if (nonEmptyText(next.instructions) === undefined) {
    next.instructions = DEFAULT_SYSTEM_PROMPT;
    changed = true;
  }
  return { payload: next as T, changed };
}

export type FreeTierGateOutcome =
  | { readonly status: "gated"; readonly body: string }
  | { readonly status: "unchanged" }
  | { readonly status: "incomplete" };

export function gateOpenCodeFreeBody(bodyText: string | undefined): FreeTierGateOutcome {
  if (!bodyText || bodyText.length > GATE_BODY_MAX_CHARS) {
    return { status: "incomplete" };
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(bodyText);
  } catch {
    return { status: "incomplete" };
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    return { status: "incomplete" };
  }
  const { payload, changed } = withOpenCodeFreeGate(parsed);
  if (!hasMarkerTools(payload as Record<string, unknown>)) {
    return { status: "incomplete" };
  }
  if (!changed) return { status: "unchanged" };
  return { status: "gated", body: JSON.stringify(payload) };
}

function readRequestUrl(input: Parameters<FetchFunction>[0]): string | undefined {
  try {
    if (input instanceof Request) return input.url;
    if (input instanceof URL) return input.href;
    return new URL(String(input)).href;
  } catch {
    return undefined;
  }
}

async function readInputBodyText(
  input: Parameters<FetchFunction>[0],
  init: Parameters<FetchFunction>[1],
): Promise<string | undefined> {
  try {
    const body = init?.body;
    if (typeof body === "string") return body;
    if (body instanceof Uint8Array) {
      return body.length > GATE_BODY_MAX_CHARS
        ? undefined
        : new TextDecoder().decode(body);
    }
    if (body instanceof ArrayBuffer) {
      return body.byteLength > GATE_BODY_MAX_CHARS
        ? undefined
        : new TextDecoder().decode(body);
    }
    if (input instanceof Request && body === undefined) {
      const text = await input.clone().text();
      return text.length > GATE_BODY_MAX_CHARS ? undefined : text;
    }
  } catch {
    // Probe failures always pass through; gating must never block a request.
  }
  return undefined;
}

/**
 * A `fetch` wrapper that completes the free-tier signature on request bodies.
 * Non-Zen hosts, non-`-free` models, and bodies the gate cannot complete pass
 * through untouched (same URL, same init) so failures stay attributable to
 * the gateway, not to a half-signed request.
 */
export function createOpenCodeFreeGateFetch(
  fetchFn?: FetchFunction,
): FetchFunction {
  const base = (fetchFn ?? globalThis.fetch.bind(globalThis)) as FetchFunction;
  return (async (input, init) => {
    const requestUrl = readRequestUrl(input);
    if (requestUrl === undefined || !isOpenCodeZenRequest(requestUrl)) {
      return base(input, init);
    }
    const bodyText = await readInputBodyText(input, init);
    const probe = bodyText?.slice(0, MODEL_PROBE_MAX_CHARS) ?? "";
    const modelId = MODEL_FIELD_PATTERN.exec(probe)?.[1];
    if (!isOpenCodeFreeTierModelId(modelId)) {
      return base(input, init);
    }
    const gated = gateOpenCodeFreeBody(bodyText);
    if (gated.status !== "gated") {
      return base(input, init);
    }
    return base(requestUrl, {
      ...init,
      method: init?.method ?? (input instanceof Request ? input.method : "POST"),
      body: gated.body,
      signal: init?.signal ?? (input instanceof Request ? input.signal : undefined),
    });
  }) as FetchFunction;
}
