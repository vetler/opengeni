import { expect, test } from "bun:test";
import type { ResolvedModelProvider } from "@opengeni/config";
import { modelRequestPolicyForProvider } from "../src/model-provider-request-policy";

const policy = modelRequestPolicyForProvider({
  id: "registry",
  label: "Registry",
  kind: "api-key",
  api: "chat",
  builtin: false,
} as ResolvedModelProvider);

const signature = { google: { thought_signature: "opaque-signature-fixture" } };

function chatBody(model: string, reasoningEffort?: string) {
  return {
    model,
    ...(reasoningEffort ? { reasoning_effort: reasoningEffort } : {}),
    messages: [
      { role: "user", content: "Look it up" },
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
          { id: "call-2", type: "function", function: { name: "lookup", arguments: "{}" } },
        ],
      },
      { role: "tool", tool_call_id: "call-1", content: "ok" },
      { role: "tool", tool_call_id: "call-2", content: "ok" },
    ],
  };
}

test("Gemini Chat requests clamp efforts Gemini rejects to high", () => {
  for (const effort of ["xhigh", "max"]) {
    const body = chatBody("gemini-3.8-flash", effort);
    const before = JSON.stringify(body);
    expect(policy({ path: "/chat/completions", body })?.body?.reasoning_effort).toBe("high");
    expect(JSON.stringify(body)).toBe(before);
  }
  for (const effort of ["none", "minimal", "low", "medium", "high"]) {
    const result = policy({
      path: "/chat/completions",
      body: chatBody("gemini-3.8-flash", effort),
    });
    expect(result?.body?.reasoning_effort ?? effort).toBe(effort);
  }
});

test("Gemini Chat requests keep the thought signature on its tool call", () => {
  const body = chatBody("google/gemini-3.8-flash");
  const sent = policy({ path: "/chat/completions", body })?.body ?? body;
  expect(sent.messages[1].tool_calls[0].extra_content).toEqual(signature);
});

test("other Chat models receive Gemini history without extra_content", () => {
  const body = chatBody("accounts/fireworks/models/glm-5p2", "xhigh");
  const before = JSON.stringify(body);
  const sent = policy({ path: "/chat/completions", body })?.body;
  expect(sent?.reasoning_effort).toBe("xhigh");
  expect(sent?.messages[1].tool_calls).toEqual([
    {
      id: "call-1",
      type: "function",
      function: { name: "lookup", arguments: "{}" },
      vendor_extension: "kept",
    },
    { id: "call-2", type: "function", function: { name: "lookup", arguments: "{}" } },
  ]);
  expect(JSON.stringify(body)).toBe(before);
});

test("Responses requests are outside the Chat-only rules", () => {
  const body = { model: "gemini-3.8-flash", reasoning: { effort: "xhigh" }, input: [] };
  expect(policy({ path: "/responses", body })?.body?.reasoning ?? body.reasoning).toEqual({
    effort: "xhigh",
  });
});
