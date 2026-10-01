/**
 * OpenCode Go (and OpenCode Zen) require a stable conversation header so the
 * gateway can pin a chat to one backend. pi-ai does not emit
 * `x-opencode-session`; the official Pi coding-agent injects it in the agent
 * layer, and this runtime does the same.
 *
 * Free-tier requests must be byte-equivalent to the native opencode CLI or the
 * gateway answers `403 FreeTierError`: timestamp-encoded `ses_`/`msg_` ids
 * (random/UUID ids are rejected even when everything else is perfect),
 * `x-opencode-client: cli` and `User-Agent: opencode/<real-version>`.
 */

import type { Api, Model, ProviderHeaders, SimpleStreamOptions } from "@earendil-works/pi-ai";
import { OPENCODE_GO_API_STYLE } from "@pi-desktop/shared";
import { createOpenCodeFreeGateFetch } from "./opencode-free-gate.js";
import type { RuntimeProviderConfig } from "./provider-binding.js";

export const OPENCODE_SESSION_HEADER = "x-opencode-session";
export const OPENCODE_REQUEST_HEADER = "x-opencode-request";
export const OPENCODE_CLIENT_HEADER = "x-opencode-client";
export const OPENCODE_CLIENT_VALUE = "cli";
// Native client fingerprint the gateway validates ("1.18.0 or newer is
// required"). Bump alongside real opencode releases:
// https://github.com/sst/opencode/releases
export const OPENCODE_CLI_VERSION = "1.18.32";
export const OPENCODE_USER_AGENT = `opencode/${OPENCODE_CLI_VERSION}`;

const SESSION_ID_PREFIX = "ses_";
const REQUEST_ID_PREFIX = "msg_";
const TIMESTAMP_HEX_LENGTH = 12;
const RANDOM_TAIL_LENGTH = 14;
const TIME_BYTES = 6;

const BASE62_CHARS =
  "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

let lastIdTimestamp = 0;
let idCounter = 0;

/**
 * Exact port of opencode `identifier.ts` descending(): `~(ms * 0x1000 +
 * per-ms counter)`, low 48 bits as lowercase hex. BigInt keeps the bitwise-NOT
 * semantics for large timestamps. Same-era ids share high bytes — if yours
 * don't, the port is wrong.
 */
export function descendingTimestampHex(now: number = Date.now()): string {
  const timestamp = Math.trunc(now);
  if (timestamp !== lastIdTimestamp) {
    lastIdTimestamp = timestamp;
    idCounter = 0;
  }
  idCounter += 1;
  const value = ~(BigInt(timestamp) * 0x1000n + BigInt(idCounter));
  let time = "";
  for (let index = 0; index < TIME_BYTES; index += 1) {
    time += Number((value >> BigInt(40 - 8 * index)) & 0xffn)
      .toString(16)
      .padStart(2, "0");
  }
  return time;
}

function randomBase62(length: number): string {
  const bytes = new Uint8Array(length);
  globalThis.crypto.getRandomValues(bytes);
  let result = "";
  for (const byte of bytes) {
    result += BASE62_CHARS[byte % BASE62_CHARS.length];
  }
  return result;
}

/** Fresh timestamp-encoded conversation id (`ses_` + 26 chars). */
export function newOpenCodeSessionId(now: number = Date.now()): string {
  return `${SESSION_ID_PREFIX}${descendingTimestampHex(now)}${randomBase62(RANDOM_TAIL_LENGTH)}`;
}

/** Fresh timestamp-encoded per-request id (`msg_` + 26 chars). */
export function newOpenCodeRequestId(now: number = Date.now()): string {
  return `${REQUEST_ID_PREFIX}${descendingTimestampHex(now)}${randomBase62(RANDOM_TAIL_LENGTH)}`;
}

function isStampedId(value: unknown, prefix: string): boolean {
  if (typeof value !== "string") return false;
  if (
    value.length !==
    prefix.length + TIMESTAMP_HEX_LENGTH + RANDOM_TAIL_LENGTH
  ) {
    return false;
  }
  if (!value.startsWith(prefix)) return false;
  return /^[0-9a-f]{12}[A-Za-z0-9]{14}$/.test(value.slice(prefix.length));
}

export function isOpenCodeSessionId(value: unknown): boolean {
  return isStampedId(value, SESSION_ID_PREFIX);
}

export function isOpenCodeRequestId(value: unknown): boolean {
  return isStampedId(value, REQUEST_ID_PREFIX);
}

const MAX_CACHED_SESSIONS = 500;
const sessionCache = new Map<string, string>();

/**
 * Stable wire session id for one harness conversation. The seed (the harness
 * session id) only selects the cache slot — the content is always a fresh
 * timestamp-encoded id generated on first sight, because the gateway pins a
 * conversation to one backend by this id. An empty seed (ad-hoc calls with no
 * conversation context) gets a fresh id per call.
 */
export function deriveOpenCodeSessionId(seed: string): string {
  const key = seed.trim();
  if (!key) return newOpenCodeSessionId(Date.now());
  let id = sessionCache.get(key);
  if (!id) {
    if (sessionCache.size >= MAX_CACHED_SESSIONS) {
      sessionCache.clear();
    }
    id = newOpenCodeSessionId(Date.now());
    sessionCache.set(key, id);
  }
  return id;
}

export type OpenCodeEndpointInput = {
  apiStyle?: string;
  vendorKey?: string;
  baseUrl?: string;
  model?: Model<Api>;
};

function hostnameOf(url: string | undefined): string | undefined {
  if (!url?.trim()) return undefined;
  try {
    return new URL(url).hostname.toLowerCase();
  } catch {
    return undefined;
  }
}

function isOpenCodeHost(url: string | undefined): boolean {
  const host = hostnameOf(url);
  return host === "opencode.ai" || (host?.endsWith(".opencode.ai") ?? false);
}

function headerValue(
  headers: ProviderHeaders | undefined,
  name: string,
): string | undefined {
  if (!headers) return undefined;
  const found = Object.entries(headers).find(
    ([key]) => key.toLowerCase() === name.toLowerCase(),
  );
  if (!found) return undefined;
  const value = found[1];
  return typeof value === "string" && value.trim() ? value : undefined;
}

export function isOpenCodeEndpoint(input: OpenCodeEndpointInput): boolean {
  if (input.apiStyle === OPENCODE_GO_API_STYLE) return true;
  const vendor = (input.vendorKey ?? "").trim().toLowerCase();
  if (vendor === "opencode" || vendor === "opencode-go") return true;
  const providerId = (input.model?.provider ?? "").trim().toLowerCase();
  if (providerId === "opencode" || providerId === "opencode-go") return true;
  return isOpenCodeHost(input.baseUrl) || isOpenCodeHost(input.model?.baseUrl);
}

export function openCodeEndpointFromProvider(
  provider: RuntimeProviderConfig,
  model?: Model<Api>,
): OpenCodeEndpointInput {
  return {
    apiStyle: provider.apiStyle,
    vendorKey: provider.vendorKey,
    baseUrl: provider.baseUrl,
    model,
  };
}

/** Merge OpenCode routing headers. An explicit caller session/request header
 * wins, except an empty/null value is replaced so the gateway cannot 400. The
 * client identity is always forced to the native CLI fingerprint — a stale
 * third-party value would fail the free-tier signature with 403. */
export function mergeOpenCodeSessionHeaders(
  input: OpenCodeEndpointInput & {
    sessionId?: string;
    requestId?: string;
    headers?: ProviderHeaders;
  },
): ProviderHeaders | undefined {
  const sessionId = input.sessionId?.trim() || undefined;
  if (!sessionId || !isOpenCodeEndpoint(input)) {
    return input.headers;
  }
  const requestId = input.requestId?.trim() || newOpenCodeRequestId();

  const injected: ProviderHeaders = {
    [OPENCODE_SESSION_HEADER]: sessionId,
    [OPENCODE_REQUEST_HEADER]: requestId,
    [OPENCODE_CLIENT_HEADER]: OPENCODE_CLIENT_VALUE,
    "User-Agent": OPENCODE_USER_AGENT,
  };
  const merged: ProviderHeaders = {
    ...injected,
    ...input.headers,
  };
  if (!headerValue(merged, OPENCODE_SESSION_HEADER)) {
    merged[OPENCODE_SESSION_HEADER] = sessionId;
  }
  if (!headerValue(merged, OPENCODE_REQUEST_HEADER)) {
    merged[OPENCODE_REQUEST_HEADER] = requestId;
  }
  merged[OPENCODE_CLIENT_HEADER] = OPENCODE_CLIENT_VALUE;
  const userAgent = headerValue(merged, "user-agent");
  if (!userAgent || !userAgent.toLowerCase().startsWith("opencode/")) {
    merged["User-Agent"] = OPENCODE_USER_AGENT;
  }
  return merged;
}

/** Attach OpenCode routing headers to a pi-ai stream options object.
 * The harness session id stays on the options for request routing; the wire
 * header carries a stable timestamp-encoded id derived from it (the gateway
 * pins a conversation to one backend by this id and rejects UUID/random ids).
 * OpenCode requests without a caller session id get a fresh stamped id per
 * call; retries reuse the same options object, so the request id they already
 * carry is preserved and the retry stays byte-identical. */
export function withOpenCodeSessionHeaders(
  options: SimpleStreamOptions | undefined,
  input: OpenCodeEndpointInput & { sessionId?: string },
): SimpleStreamOptions {
  const preferred =
    (options?.sessionId ?? input.sessionId)?.trim() || undefined;
  const onOpenCode = isOpenCodeEndpoint(input);
  const sessionId =
    preferred ?? (onOpenCode ? newOpenCodeSessionId() : undefined);
  const wireSessionId =
    onOpenCode && sessionId
      ? (preferred ? deriveOpenCodeSessionId(preferred) : sessionId)
      : sessionId;
  const headers = mergeOpenCodeSessionHeaders({
    ...input,
    sessionId: wireSessionId,
    headers: options?.headers,
  });
  // The body gate rides the request's fetch so every call site already wrapped
  // here (session turns, subagents, one-shot, compaction) shares one hook.
  const gateFetch =
    onOpenCode && sessionId
      ? createOpenCodeFreeGateFetch(options?.fetch)
      : options?.fetch;
  return {
    ...(options ?? {}),
    ...(sessionId ? { sessionId } : {}),
    ...(headers ? { headers } : {}),
    ...(gateFetch !== undefined && gateFetch !== options?.fetch
      ? { fetch: gateFetch }
      : {}),
  };
}
