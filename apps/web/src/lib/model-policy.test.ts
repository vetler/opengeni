import { expect, test } from "bun:test";
import type { ClientModel, ModelCapabilitiesV1 } from "@opengeni/sdk";
import { reasoningEffortAllowedForModel } from "./model-policy";

function model(reasoning?: Partial<ModelCapabilitiesV1["reasoning"]>): ClientModel {
  return {
    id: "fixture",
    label: "Fixture",
    provider: "fixture",
    providerLabel: "Fixture",
    api: "chat",
    ...(reasoning
      ? {
          capabilities: {
            reasoning: {
              upstream: "supported",
              runnable: true,
              efforts: [],
              defaultEffort: null,
              required: false,
              ...reasoning,
            },
          } as ModelCapabilitiesV1,
        }
      : {}),
  } as ClientModel;
}

test("a model with effort control allows only its own efforts", () => {
  const reasoning = model({ efforts: ["low", "medium", "high"], defaultEffort: "medium" });
  expect(reasoningEffortAllowedForModel(reasoning, "medium")).toBe(true);
  expect(reasoningEffortAllowedForModel(reasoning, "xhigh")).toBe(false);
});

test("a model without effort control allows any accepted effort", () => {
  // The server default for such a model is the deployment effort (here xhigh),
  // not the picker's `low` placeholder; neither reaches the provider.
  for (const noControl of [model({ upstream: "unknown", runnable: false }), model()]) {
    expect(reasoningEffortAllowedForModel(noControl, "xhigh")).toBe(true);
    expect(reasoningEffortAllowedForModel(noControl, "low")).toBe(true);
  }
});
