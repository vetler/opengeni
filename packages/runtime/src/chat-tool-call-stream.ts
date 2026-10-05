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
 *   each step, and to the message of a text answer, and rejects the next
 *   request with `400 Function call is missing a thought_signature` unless a
 *   call's object is replayed on the same call. `withChatToolCallExtraContent`
 *   restores both to the completed items' `providerData`, which the SDK
 *   spreads back onto the replayed tool call and assistant message.
 */
function object(value: unknown): JsonObject | undefined {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonObject)
    : undefined;
}

type ToolCallKind = "function" | "custom";

type ToolCallIndexes = {
  byCallId: Map<string, number>;
  argumentsByIndex: Map<number, string>;
  /** The latest call of each kind; continuation deltas follow their own kind. */
  last: Partial<Record<ToolCallKind, number>>;
  next: number;
};

function isCompleteJson(text: string | undefined): boolean {
  if (!text) return false;
  try {
    JSON.parse(text);
    return true;
  } catch {
    return false;
  }
}

/** An index-less delta starts a new call when it names an unseen call id and a
 * function or custom tool, or when it carries an unseen id after the previous
 * call of its kind has complete arguments; anything else continues that call. */
function withToolCallIndexes(delta: JsonObject, indexes: ToolCallIndexes): JsonObject {
  if (!Array.isArray(delta.tool_calls)) return delta;
  let changed = false;
  const toolCalls = delta.tool_calls.map((value: unknown) => {
    const call = object(value);
    if (!call) return value;
    const kind: ToolCallKind =
      call.type === "custom" || object(call.custom) ? "custom" : "function";
    let index = typeof call.index === "number" ? call.index : undefined;
    const indexed = index !== undefined;
    if (index === undefined) {
      const id = typeof call.id === "string" && call.id.length > 0 ? call.id : undefined;
      const name = object(call[kind])?.name;
      const named = typeof name === "string" && name.length > 0;
      const previous = indexes.last[kind];
      index = id !== undefined ? indexes.byCallId.get(id) : undefined;
      if (index === undefined) {
        const previousComplete =
          previous === undefined || isCompleteJson(indexes.argumentsByIndex.get(previous));
        const startsCall = id !== undefined && (named || previousComplete);
        index = startsCall || previous === undefined ? indexes.next++ : previous;
        if (startsCall && id !== undefined) indexes.byCallId.set(id, index);
      }
    } else {
      indexes.next = Math.max(indexes.next, index + 1);
    }
    const fragment = object(call.function)?.arguments;
    if (typeof fragment === "string") {
      indexes.argumentsByIndex.set(index, (indexes.argumentsByIndex.get(index) ?? "") + fragment);
    }
    indexes.last[kind] = index;
    if (indexed) return value;
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
      if (!state) {
        state = { byCallId: new Map(), argumentsByIndex: new Map(), last: {}, next: 0 };
        indexes.set(choice.index, state);
      }
      const indexed = withToolCallIndexes(delta, state);
      if (indexed === delta) return value;
      changed = true;
      return { ...choice, delta: indexed };
    });
    yield changed ? ({ ...(chunk as JsonObject), choices: projected } as T) : chunk;
  }
}

/** Signatures by the SDK's call id (the first id seen for an index, read from
 * chunks `indexChatToolCallChunks` already indexed) and for the message. */
export type ChatToolCallExtraContent = {
  byCallId: Map<string, JsonObject>;
  callIdByIndex: Map<number, string>;
  message?: JsonObject;
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
  const record = object(delta);
  const messageExtraContent = object(record?.extra_content);
  if (messageExtraContent) {
    accumulated.message = mergeExtraContent(accumulated.message, messageExtraContent);
  }
  const toolCalls = record?.tool_calls;
  if (!Array.isArray(toolCalls)) return;
  for (const value of toolCalls) {
    const call = object(value);
    if (!call || typeof call.index !== "number") continue;
    const id = typeof call.id === "string" && call.id.length > 0 ? call.id : undefined;
    const callId = accumulated.callIdByIndex.get(call.index) ?? id;
    if (callId === undefined) continue;
    accumulated.callIdByIndex.set(call.index, callId);
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
  if (accumulated.byCallId.size === 0 && !accumulated.message) return output;
  let messageAttached = false;
  return output.map((item) => {
    if (item.type === "function_call") {
      const extraContent = accumulated.byCallId.get(item.callId);
      if (!extraContent) return item;
      return {
        ...item,
        providerData: { ...item.providerData, extra_content: structuredClone(extraContent) },
      };
    }
    if (
      item.type === "message" &&
      item.role === "assistant" &&
      accumulated.message &&
      !messageAttached
    ) {
      messageAttached = true;
      return {
        ...item,
        providerData: { ...item.providerData, extra_content: structuredClone(accumulated.message) },
      };
    }
    return item;
  });
}
