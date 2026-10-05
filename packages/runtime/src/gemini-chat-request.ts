import type { ConfiguredModel, ResolvedModelProvider } from "@opengeni/config";
import { ReasoningEffort } from "@opengeni/contracts";
import { isGeminiUpstreamModel } from "./gemini-function-response";
import type { ModelJsonRequestPolicy } from "./replayable-json-body";

/**
 * Gemini Chat Completions request quirks (request-local only).
 *
 * - Reasoning effort follows the model's declared catalog efforts. A model
 *   without runnable reasoning control sends none, so Gemini uses its own
 *   default thinking level instead of the deployment effort. Otherwise the
 *   effort clamps to the highest declared level at or below the requested
 *   one. Every Gemini 3 model rejects `minimal`, `xhigh` and `max`, so those
 *   never count as declared; per-model limits (Pro also rejects `none`) come
 *   from the declared efforts.
 * - `extra_content` carries Gemini thought signatures on assistant messages
 *   and tool calls, and only a direct Gemini route reads it. Other Chat routes,
 *   including OpenRouter and Vercel Gateway, which carry their own reasoning
 *   state, receive the history without it.
 *
 * Canonical history is unchanged; only the outgoing request body is copied.
 */
export type ModelReasoningLookup = ReadonlyMap<
  string,
  ConfiguredModel["capabilities"]["reasoning"]
>;

const GEMINI_REJECTED_EFFORTS = new Set<string>(["minimal", "xhigh", "max"]);
const EFFORT_ORDER: readonly string[] = ReasoningEffort.options;

function isGoogleEndpoint(baseUrl: string | undefined): boolean {
  if (!baseUrl) return false;
  try {
    const host = new URL(baseUrl).hostname;
    return (
      host === "generativelanguage.googleapis.com" || host.endsWith("aiplatform.googleapis.com")
    );
  } catch {
    return false;
  }
}

function isRoutedProvider(provider: ResolvedModelProvider): boolean {
  return provider.kind.startsWith("openrouter-") || provider.kind.startsWith("vercel-gateway-");
}

/** The effort to send to Gemini, or undefined to send none. */
function geminiReasoningEffort(
  reasoning: ConfiguredModel["capabilities"]["reasoning"] | undefined,
  requested: string,
): string | undefined {
  if (!reasoning?.runnable) return undefined;
  const supported = EFFORT_ORDER.filter(
    (effort) =>
      reasoning.efforts.includes(effort as ReasoningEffort) && !GEMINI_REJECTED_EFFORTS.has(effort),
  );
  if (supported.includes(requested)) return requested;
  const ceiling = EFFORT_ORDER.indexOf(requested);
  const below = supported.filter((effort) => EFFORT_ORDER.indexOf(effort) <= ceiling);
  const fallback =
    reasoning.defaultEffort && supported.includes(reasoning.defaultEffort)
      ? reasoning.defaultEffort
      : supported[0];
  return below.at(-1) ?? fallback;
}

function withoutExtraContent(messages: unknown[]): unknown[] {
  let changed = false;
  const projected = messages.map((message) => {
    if (!message || typeof message !== "object" || Array.isArray(message)) return message;
    const record = message as Record<string, unknown>;
    if (record.role !== "assistant") return message;
    let next = record;
    if ("extra_content" in record) {
      const { extra_content: _geminiOnly, ...rest } = record;
      next = rest;
    }
    if (Array.isArray(record.tool_calls)) {
      let callsChanged = false;
      const toolCalls = record.tool_calls.map((call: unknown) => {
        if (!call || typeof call !== "object" || !("extra_content" in call)) return call;
        callsChanged = true;
        const { extra_content: _geminiOnly, ...rest } = call as Record<string, unknown>;
        return rest;
      });
      if (callsChanged) next = { ...next, tool_calls: toolCalls };
    }
    if (next === record) return message;
    changed = true;
    return next;
  });
  return changed ? projected : messages;
}

export function geminiChatRequestPolicy(
  provider: ResolvedModelProvider,
  modelReasoning?: ModelReasoningLookup,
): ModelJsonRequestPolicy {
  return ({ path, body }) => {
    if (!(path.split("?", 1)[0] ?? path).endsWith("/chat/completions")) return undefined;
    const geminiRoute =
      !isRoutedProvider(provider) &&
      (isGoogleEndpoint(provider.baseUrl) || isGeminiUpstreamModel(body.model));
    if (!geminiRoute) {
      if (!Array.isArray(body.messages)) return undefined;
      const messages = withoutExtraContent(body.messages);
      return messages === body.messages ? undefined : { body: { ...body, messages } };
    }
    if (typeof body.reasoning_effort !== "string") return undefined;
    const effort = geminiReasoningEffort(
      typeof body.model === "string" ? modelReasoning?.get(body.model) : undefined,
      body.reasoning_effort,
    );
    if (effort === body.reasoning_effort) return undefined;
    const { reasoning_effort: _requested, ...rest } = body;
    return { body: effort === undefined ? rest : { ...rest, reasoning_effort: effort } };
  };
}
