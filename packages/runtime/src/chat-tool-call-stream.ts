import type { ModelResponse } from "@openai/agents";

type JsonObject = Record<string, unknown>;

/**
 * Streamed Chat Completions tool calls that the SDK accumulator mishandles.
 *
 * The SDK builds streamed calls keyed by each delta's `index` and keeps only
 * id/name/arguments. Google's OpenAI-compatible endpoint differs on both:
 *
 * - It streams each call whole, with its own id and no `index`. The SDK files
 *   every such delta under one `undefined` key, so parallel calls merge into a
 *   single call with concatenated names and arguments.
 * - It attaches `extra_content.google.thought_signature` to the first call of
 *   each step and rejects the next request with `400 Function call is missing
 *   a thought_signature` unless that object is replayed on the same call.
 *
 * This accumulator follows the raw deltas by call id (falling back to an index
 * seen earlier in the response for OpenAI-style continuation deltas), then
 * repairs the SDK's completed output: merged index-less calls are rebuilt as
 * separate calls, and `extra_content` is restored to `providerData`, matching
 * the SDK's non-streamed shape so it replays the field on that tool call.
 * OpenAI-style indexed streams keep the SDK's own items.
 */
type StreamedToolCall = {
  callId: string;
  name: string;
  arguments: string;
  extraContent?: JsonObject;
};

export type ChatToolCallStream = {
  calls: StreamedToolCall[];
  byCallId: Map<string, StreamedToolCall>;
  callIdByIndex: Map<number, string>;
  sawUnindexed: boolean;
};

function object(value: unknown): JsonObject | undefined {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonObject)
    : undefined;
}

export function newChatToolCallStream(): ChatToolCallStream {
  return { calls: [], byCallId: new Map(), callIdByIndex: new Map(), sawUnindexed: false };
}

export function appendChatToolCallDeltas(stream: ChatToolCallStream, delta: unknown): void {
  const toolCalls = object(delta)?.tool_calls;
  if (!Array.isArray(toolCalls)) return;
  for (const value of toolCalls) {
    const call = object(value);
    // The SDK ignores custom tool calls; so does this accumulator.
    if (!call || call.type === "custom") continue;
    const index = typeof call.index === "number" ? call.index : undefined;
    if (index === undefined) stream.sawUnindexed = true;
    const id = typeof call.id === "string" && call.id.length > 0 ? call.id : undefined;
    const callId =
      id ?? (index !== undefined ? stream.callIdByIndex.get(index) : stream.calls.at(-1)?.callId);
    if (callId === undefined) continue;
    if (index !== undefined) stream.callIdByIndex.set(index, callId);
    let target = stream.byCallId.get(callId);
    if (!target) {
      target = { callId, name: "", arguments: "" };
      stream.byCallId.set(callId, target);
      stream.calls.push(target);
    }
    const fn = object(call.function);
    if (typeof fn?.name === "string") target.name += fn.name;
    if (typeof fn?.arguments === "string") target.arguments += fn.arguments;
    const extraContent = object(call.extra_content);
    if (extraContent)
      target.extraContent = { ...target.extraContent, ...structuredClone(extraContent) };
  }
}

export function withChatToolCallStream<T extends ModelResponse["output"][number]>(
  output: T[],
  stream: ChatToolCallStream,
): T[] {
  let projected = output;
  const sdkCalls = output.filter((item) => item.type === "function_call");
  const template = sdkCalls[0];
  if (template && stream.sawUnindexed && stream.calls.length > sdkCalls.length) {
    const rebuilt = stream.calls.map(
      (call) =>
        ({
          ...template,
          callId: call.callId,
          name: call.name,
          // Same workaround the SDK applies to its own accumulated arguments.
          arguments: call.arguments.startsWith("{}{") ? call.arguments.slice(2) : call.arguments,
        }) as T,
    );
    projected = [...output.filter((item) => item.type !== "function_call"), ...rebuilt];
  }
  if (!stream.calls.some((call) => call.extraContent)) return projected;
  return projected.map((item) => {
    if (item.type !== "function_call") return item;
    const extraContent = stream.byCallId.get(item.callId)?.extraContent;
    if (!extraContent) return item;
    return {
      ...item,
      providerData: { ...item.providerData, extra_content: structuredClone(extraContent) },
    };
  });
}
