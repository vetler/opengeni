import { expect, test } from "bun:test";
import type { ConfiguredModel, ResolvedModelProvider } from "@opengeni/config";
import type { ModelReasoningLookup } from "../src/gemini-chat-request";
import { modelRequestPolicyForProvider } from "../src/model-provider-request-policy";

type Reasoning = ConfiguredModel["capabilities"]["reasoning"];

function provider(overrides: Partial<ResolvedModelProvider> = {}): ResolvedModelProvider {
  return {
    id: "gemini",
    label: "Google Gemini",
    kind: "api-key",
    api: "chat",
    builtin: false,
    baseUrl: "https://generativelanguage.googleapis.com/v1beta/openai",
    ...overrides,
  } as ResolvedModelProvider;
}

function reasoning(
  efforts: Reasoning["efforts"],
  defaultEffort: Reasoning["defaultEffort"] = null,
) {
  return {
    upstream: "supported",
    runnable: true,
    efforts,
    defaultEffort,
    required: false,
  } as Reasoning;
}

const lookup: ModelReasoningLookup = new Map<string, Reasoning>([
  ["gemini-no-control", { ...reasoning([]), upstream: "unknown", runnable: false }],
  ["gemini-legacy-list", reasoning(["low", "medium", "high", "xhigh", "max"], "xhigh")],
  ["gemini-pro-like", reasoning(["low", "medium", "high"], "high")],
  ["gemini-flash-like", reasoning(["none", "low", "medium", "high"], "medium")],
]);

const signature = { google: { thought_signature: "opaque-signature-fixture" } };

function chatBody(model: string, reasoningEffort?: string) {
  return {
    model,
    ...(reasoningEffort ? { reasoning_effort: reasoningEffort } : {}),
    messages: [
      { role: "user", content: "Look it up" },
      { role: "assistant", content: "Thinking it through.", extra_content: signature },
      {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: "call-1",
            type: "function",
            function: { name: "lookup", arguments: "{}" },
            extra_content: signature,
            vendor_extension: "kept",
          },
        ],
      },
      { role: "tool", tool_call_id: "call-1", content: "ok" },
    ],
  };
}

function sent(policyProvider: ResolvedModelProvider, body: Record<string, any>) {
  const before = JSON.stringify(body);
  const result = modelRequestPolicyForProvider(
    policyProvider,
    undefined,
    lookup,
  )({
    path: "/chat/completions",
    body,
  });
  expect(JSON.stringify(body)).toBe(before);
  return (result?.body ?? body) as Record<string, any>;
}

test("Gemini sends no effort for a model without declared effort control", () => {
  for (const model of ["gemini-no-control", "gemini-not-configured"]) {
    expect("reasoning_effort" in sent(provider(), chatBody(model, "xhigh"))).toBe(false);
  }
});

test("Gemini clamps to the declared efforts Gemini accepts", () => {
  const cases: Array<[string, string, string]> = [
    // A legacy deployment list still drops the levels every Gemini model rejects.
    ["gemini-legacy-list", "xhigh", "high"],
    ["gemini-legacy-list", "max", "high"],
    ["gemini-legacy-list", "minimal", "low"],
    ["gemini-legacy-list", "medium", "medium"],
    // Pro rejects `none`. Nothing is declared at or below it, so the model's
    // default applies, as in core's clampReasoningEffortForConfiguredModel.
    ["gemini-pro-like", "none", "high"],
    ["gemini-pro-like", "xhigh", "high"],
    ["gemini-flash-like", "none", "none"],
    ["gemini-flash-like", "minimal", "none"],
  ];
  for (const [model, requested, expected] of cases) {
    expect([
      model,
      requested,
      sent(provider(), chatBody(model, requested)).reasoning_effort,
    ]).toEqual([model, requested, expected]);
  }
});

test("a Google route keeps thought signatures, even without gemini in the model id", () => {
  for (const model of ["gemini-flash-like", "learnlm-2.0-flash"]) {
    const body = sent(provider(), chatBody(model));
    expect(body.messages[1].extra_content).toEqual(signature);
    expect(body.messages[2].tool_calls[0].extra_content).toEqual(signature);
  }
});

test("other Chat routes receive Gemini history without extra_content", () => {
  const routes: Array<[ResolvedModelProvider, string]> = [
    [provider({ id: "fireworks", baseUrl: "https://api.fireworks.ai/inference/v1" }), "glm-5p2"],
    [
      provider({
        id: "workspace-openrouter",
        kind: "openrouter-workspace",
        baseUrl: "https://openrouter.ai/api/v1",
      }),
      "google/gemini-3.8-flash",
    ],
  ];
  for (const [route, model] of routes) {
    const body = sent(route, chatBody(model, "xhigh"));
    expect(body.reasoning_effort).toBe("xhigh");
    expect(body.messages[1]).toEqual({ role: "assistant", content: "Thinking it through." });
    expect(body.messages[2].tool_calls).toEqual([
      {
        id: "call-1",
        type: "function",
        function: { name: "lookup", arguments: "{}" },
        vendor_extension: "kept",
      },
    ]);
  }
});

test("a non-streamed reply's signature moves from its text part to the message", () => {
  const body = {
    model: "gemini-flash-like",
    messages: [
      { role: "user", content: "Is 1001 prime?" },
      {
        role: "assistant",
        content: [{ type: "text", text: "No.", role: "assistant", extra_content: signature }],
      },
    ],
  };
  const message = sent(provider(), body).messages[1];
  expect(message.content).toEqual([{ type: "text", text: "No." }]);
  expect(message.extra_content).toEqual(signature);
});

test("Responses requests are outside the Chat-only rules", () => {
  const body = { model: "gemini-flash-like", reasoning: { effort: "xhigh" }, input: [] };
  const result = modelRequestPolicyForProvider(
    provider(),
    undefined,
    lookup,
  )({
    path: "/responses",
    body,
  });
  expect(result?.body?.reasoning ?? body.reasoning).toEqual({ effort: "xhigh" });
});
