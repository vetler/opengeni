import type { ModelResponse } from "@openai/agents";

type JsonObject = Record<string, unknown>;

/**
 * Streamed Chat Completions tool calls in Google's OpenAI-compatible shape.
 *
 * The SDK accumulates streamed calls keyed by each delta's `index` and keeps
 * only id/name/arguments. Google differs on both:
 *
 * - It streams each call whole, with its own id and no `index`. Keyed by
 *   `undefined`, parallel calls would merge into one call with concatenated
 *   names and arguments. `indexChatToolCallChunks` gives index-less deltas an
 *   index before the SDK sees them, so it builds separate calls itself.
 * - It attaches `extra_content.google.thought_signature` to the first call of
 *   each step and rejects the next request with `400 Function call is missing
 *   a thought_signature` unless that object is replayed on the same call.
 *   `withChatToolCallExtraContent` restores it to the completed call's
 *   `providerData`, the SDK's non-streamed shape, which the SDK replays.
 */
function object(value: unknown): JsonObject | undefined {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonObject)
    : undefined;
}

type ToolCallIndexes = { byCallId: Map<string, number>; last?: number; next: number };

/** An index-less delta starts a new call only when it names a new call id and
 * a function; anything else continues the previous call. */
function withToolCallIndexes(delta: JsonObject, indexes: ToolCallIndexes): JsonObject {
  if (!Array.isArray(delta.tool_calls)) return delta;
  let changed = false;
  const toolCalls = delta.tool_calls.map((value: unknown) => {
    const call = object(value);
    if (!call) return value;
    if (typeof call.index === "number") {
      indexes.next = Math.max(indexes.next, call.index + 1);
      indexes.last = call.index;
      return value;
    }
    const id = typeof call.id === "string" && call.id.length > 0 ? call.id : undefined;
    const name = object(call.function)?.name;
    const startsCall = id !== undefined && typeof name === "string" && name.length > 0;
    let index = id !== undefined ? indexes.byCallId.get(id) : undefined;
    if (index === undefined) {
      index = startsCall || indexes.last === undefined ? indexes.next++ : indexes.last;
    }
    if (id !== undefined && startsCall) indexes.byCallId.set(id, index);
    indexes.last = index;
    changed = true;
    return { ...call, index };
  });
  return changed ? { ...delta, tool_calls: toolCalls } : delta;
}

export async function* indexChatToolCallChunks<T>(chunks: AsyncIterable<T>): AsyncIterable<T> {
  const indexes = new Map<unknown, ToolCallIndexes>();
  for await (const chunk of chunks) {
    const choices = object(chunk)?.choices;
    if (!Array.isArray(choices)) {
      yield chunk;
      continue;
    }
    let changed = false;
    const projected = choices.map((value: unknown) => {
      const choice = object(value);
      const delta = object(choice?.delta);
      if (!choice || !delta) return value;
      let state = indexes.get(choice.index);
      if (!state) indexes.set(choice.index, (state = { byCallId: new Map(), next: 0 }));
      const indexed = withToolCallIndexes(delta, state);
      if (indexed === delta) return value;
      changed = true;
      return { ...choice, delta: indexed };
    });
    yield changed ? ({ ...(chunk as JsonObject), choices: projected } as T) : chunk;
  }
}

export type ChatToolCallExtraContent = {
  byCallId: Map<string, JsonObject>;
  callIdByIndex: Map<number, string>;
};

export function newChatToolCallExtraContent(): ChatToolCallExtraContent {
  return { byCallId: new Map(), callIdByIndex: new Map() };
}

/** Later deltas extend earlier ones; nested objects merge instead of replacing. */
function mergeExtraContent(previous: JsonObject | undefined, next: JsonObject): JsonObject {
  const merged: JsonObject = { ...previous };
  for (const [key, value] of Object.entries(next)) {
    const nested = object(value);
    const existing = object(merged[key]);
    merged[key] = nested && existing ? mergeExtraContent(existing, nested) : structuredClone(value);
  }
  return merged;
}

export function appendChatToolCallExtraContent(
  accumulated: ChatToolCallExtraContent,
  delta: unknown,
): void {
  const toolCalls = object(delta)?.tool_calls;
  if (!Array.isArray(toolCalls)) return;
  for (const value of toolCalls) {
    const call = object(value);
    if (!call) continue;
    const index = typeof call.index === "number" ? call.index : undefined;
    const id = typeof call.id === "string" && call.id.length > 0 ? call.id : undefined;
    // Like the SDK, a call keeps the first id seen for its index.
    const known = index !== undefined ? accumulated.callIdByIndex.get(index) : undefined;
    const callId = known ?? id;
    if (callId === undefined) continue;
    if (index !== undefined && known === undefined) accumulated.callIdByIndex.set(index, callId);
    const extraContent = object(call.extra_content);
    if (extraContent) {
      accumulated.byCallId.set(
        callId,
        mergeExtraContent(accumulated.byCallId.get(callId), extraContent),
      );
    }
  }
}

export function withChatToolCallExtraContent<T extends ModelResponse["output"][number]>(
  output: T[],
  accumulated: ChatToolCallExtraContent,
): T[] {
  if (accumulated.byCallId.size === 0) return output;
  return output.map((item) => {
    if (item.type !== "function_call") return item;
    const extraContent = accumulated.byCallId.get(item.callId);
    if (!extraContent) return item;
    return {
      ...item,
      providerData: { ...item.providerData, extra_content: structuredClone(extraContent) },
    };
  });
}
