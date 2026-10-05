import { isGeminiUpstreamModel } from "./gemini-function-response";
import type { ModelJsonRequestPolicy } from "./replayable-json-body";

/**
 * Gemini Chat Completions request quirks (request-local only).
 *
 * - Gemini accepts `reasoning_effort` none, minimal, low, medium and high, and
 *   rejects the whole request for `xhigh` or `max`. Those clamp to `high`, the
 *   highest supported effort at or below the requested one.
 * - `tool_calls[].extra_content` carries Gemini thought signatures, which only
 *   a Gemini upstream reads. A session that switches to another Chat model
 *   sends that model its history without them.
 *
 * Canonical history is unchanged; only the outgoing request body is copied.
 */
const GEMINI_UNSUPPORTED_EFFORTS = new Set(["xhigh", "max"]);

function withoutToolCallExtraContent(messages: unknown[]): unknown[] {
  let changed = false;
  const projected = messages.map((message) => {
    if (!message || typeof message !== "object" || Array.isArray(message)) return message;
    const record = message as Record<string, unknown>;
    if (record.role !== "assistant" || !Array.isArray(record.tool_calls)) return message;
    let callsChanged = false;
    const toolCalls = record.tool_calls.map((call: unknown) => {
      if (!call || typeof call !== "object" || !("extra_content" in call)) return call;
      callsChanged = true;
      const { extra_content: _geminiOnly, ...rest } = call as Record<string, unknown>;
      return rest;
    });
    if (!callsChanged) return message;
    changed = true;
    return { ...record, tool_calls: toolCalls };
  });
  return changed ? projected : messages;
}

export const geminiChatRequestPolicy: ModelJsonRequestPolicy = ({ path, body }) => {
  if (!(path.split("?", 1)[0] ?? path).endsWith("/chat/completions")) return undefined;
  if (isGeminiUpstreamModel(body.model)) {
    return GEMINI_UNSUPPORTED_EFFORTS.has(body.reasoning_effort as string)
      ? { body: { ...body, reasoning_effort: "high" } }
      : undefined;
  }
  if (!Array.isArray(body.messages)) return undefined;
  const messages = withoutToolCallExtraContent(body.messages);
  return messages === body.messages ? undefined : { body: { ...body, messages } };
};
