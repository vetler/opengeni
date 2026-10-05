import { expect, test } from "bun:test";
import { Agent, Runner, tool } from "@openai/agents";
import { z } from "zod";
import { OpenGeniChatCompletionsModel } from "../src/model-provider-routing";
import { modelRequestPolicyForProvider } from "../src/model-provider-request-policy";
import { projectHistoryForProvider } from "../src/provider-history-adapter";
import { ReplayableJsonOpenAI, requestBodyText } from "../src/replayable-json-body";

const signature = { google: { thought_signature: "opaque-signature-fixture" } };

type Shape = "gemini" | "openai-deltas" | "fragmented-ids" | "split-extra";

const splitExtra = { google: { other: "later-fixture" } };
const mergedExtra = { google: { ...signature.google, ...splitExtra.google } };

// Gemini's compatible endpoint streams each tool call whole, in its own chunk,
// with its id and no `index`; only the first call of a step carries the
// signature. OpenAI-style streams split a call across deltas that share an
// index, and only the first carries the id. The last two shapes are edge cases:
// an index-less call whose continuation chunk carries a fresh id but no name,
// and a signature object that arrives across two deltas.
function toolCallDeltas(shape: Shape, labels: string[]) {
  return labels.flatMap((label, index) => {
    const args = JSON.stringify({ key: label });
    const call = { id: `call-${label}`, type: "function", function: { name: "lookup" } };
    const extra = index === 0 ? { extra_content: signature } : {};
    if (shape === "gemini")
      return [
        { tool_calls: [{ ...call, function: { ...call.function, arguments: args }, ...extra }] },
      ];
    if (shape === "fragmented-ids")
      return [
        {
          tool_calls: [
            { ...call, function: { name: "lookup", arguments: args.slice(0, 5) }, ...extra },
          ],
        },
        {
          tool_calls: [
            { id: `fragment-${label}`, type: "function", function: { arguments: args.slice(5) } },
          ],
        },
      ];
    const later = shape === "split-extra" && index === 0 ? { extra_content: splitExtra } : {};
    return [
      { tool_calls: [{ index, ...call, ...extra }] },
      { tool_calls: [{ index, function: { arguments: args }, ...later }] },
    ];
  });
}

const cases = [
  { shape: "gemini", labels: ["first"] },
  { shape: "gemini", labels: ["first", "second", "third"] },
  { shape: "openai-deltas", labels: ["first"] },
  { shape: "openai-deltas", labels: ["first", "second", "third"] },
  { shape: "fragmented-ids", labels: ["first"] },
  { shape: "split-extra", labels: ["first", "second"] },
] as const;

for (const { shape, labels } of cases) {
  test(`streamed tool calls replay separately with their extra_content (${shape}, ${labels.length} calls)`, async () => {
    const requests: Record<string, any>[] = [];
    let executions = 0;
    const client = new ReplayableJsonOpenAI(
      {
        apiKey: "fixture-key",
        baseURL: "https://example.test/v1beta/openai",
        maxRetries: 0,
        fetch: async (_url, init) => {
          const body = JSON.parse(await requestBodyText(init?.body));
          requests.push(body);
          const turn = requests.length;
          const common = { id: `reply-${turn}`, model: body.model, created: 1 };
          const deltas =
            turn === 1
              ? [{ role: "assistant" }, ...toolCallDeltas(shape, [...labels])]
              : [{ role: "assistant", content: "Done." }];
          // Gemini finishes a tool-call step with `stop`, not `tool_calls`.
          const toolFinish = shape === "gemini" ? "stop" : "tool_calls";
          const chunks = [
            ...deltas.map((delta) => ({ delta, finish_reason: null })),
            { delta: {}, finish_reason: turn === 1 ? toolFinish : "stop" },
          ].map((choice) => ({
            ...common,
            object: "chat.completion.chunk",
            choices: [{ index: 0, ...choice }],
          }));
          return new Response(
            chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join("") +
              "data: [DONE]\n\n",
            { headers: { "content-type": "text/event-stream" } },
          );
        },
      },
      {
        modelRequestPolicy: modelRequestPolicyForProvider({
          id: "gemini",
          label: "Google Gemini",
          kind: "api-key",
          api: "chat",
          builtin: false,
        }),
      },
    );
    const agent = new Agent({
      name: "Fixture",
      model: new OpenGeniChatCompletionsModel(client, "gemini-3.8-flash"),
      tools: [
        tool({
          name: "lookup",
          description: "Lookup",
          parameters: z.object({ key: z.string() }),
          execute: async ({ key }) => {
            executions++;
            return `${key} result`;
          },
        }),
      ],
    });
    const run = await new Runner({ tracingDisabled: true }).run(agent, "Look it up", {
      stream: true,
    });
    for await (const _event of run) {
      /* consume SDK stream */
    }
    await run.completed;

    const expectedExtra = shape === "split-extra" ? mergedExtra : signature;
    expect(executions).toBe(labels.length);
    expect(requests).toHaveLength(2);
    const replayed = requests[1]!.messages.find(
      (message: Record<string, any>) => message.role === "assistant",
    );
    expect(replayed.tool_calls).toEqual(
      labels.map((label, index) => ({
        id: `call-${label}`,
        type: "function",
        function: { name: "lookup", arguments: JSON.stringify({ key: label }) },
        ...(index === 0 ? { extra_content: expectedExtra } : {}),
      })),
    );
    const results = requests[1]!.messages.filter(
      (message: Record<string, any>) => message.role === "tool",
    );
    expect(results.map((message: Record<string, any>) => message.tool_call_id)).toEqual(
      labels.map((label) => `call-${label}`),
    );

    // Durable history is plain JSON; the signature must survive that round trip
    // so a later turn or recovered attempt replays it too.
    const persisted = JSON.parse(JSON.stringify(run.history));
    const calls = persisted.filter((item: Record<string, any>) => item.type === "function_call");
    expect(calls.map((call: Record<string, any>) => call.callId)).toEqual(
      labels.map((label) => `call-${label}`),
    );
    expect(calls[0].providerData.extra_content).toEqual(expectedExtra);
    for (const call of calls.slice(1)) expect(call.providerData?.extra_content).toBeUndefined();
  });
}

test("extra_content stays on Chat history and is dropped for other wire APIs", () => {
  const items = [
    { type: "message", role: "user", content: "Look it up" },
    {
      type: "function_call",
      callId: "call-first",
      name: "lookup",
      arguments: "{}",
      providerData: { extra_content: signature, vendor_extension: "kept" },
    },
    {
      type: "function_call",
      callId: "call-second",
      name: "lookup",
      arguments: "{}",
      providerData: { extra_content: signature },
    },
    { type: "function_call_result", callId: "call-first", output: "ok" },
  ];
  const before = JSON.stringify(items);

  expect(projectHistoryForProvider(items, "chat")).toBe(items);
  for (const api of ["responses", "anthropic-messages"] as const) {
    const projected = projectHistoryForProvider(items, api);
    expect(projected[1]).toEqual({
      type: "function_call",
      callId: "call-first",
      name: "lookup",
      arguments: "{}",
      providerData: { vendor_extension: "kept" },
    });
    expect(projected[2]).toEqual({
      type: "function_call",
      callId: "call-second",
      name: "lookup",
      arguments: "{}",
    });
  }
  expect(JSON.stringify(items)).toBe(before);
});
