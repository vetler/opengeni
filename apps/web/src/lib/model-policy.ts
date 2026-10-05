import type { ClientModel, ReasoningEffort } from "@opengeni/sdk";
import { effortOptionsForModel } from "@opengeni/react";

/** Web-facing model-picker helpers (re-exported from `@opengeni/react`). */
export {
  advancedSourceSummary,
  availabilityReasonLabel,
  billingClassForModel,
  billingClassLabel,
  coerceReasoningEffortForModel,
  effortOptionsForModel,
  findPickerRow,
  groupPickerRowsByBillingClass,
  labelLatencyMode,
  payerSummaryForModel,
  modelUsesCredits,
  projectPickerRows,
  runnableLatencyModesForModel,
  defaultEffortForModel,
  sortPickerRows,
  type PickerModelRow,
} from "@opengeni/react";

/**
 * Whether a send may carry this reasoning effort. A model whose catalog lists
 * no efforts offers no effort choice, so the session's recorded effort (the
 * server default is the deployment effort) is accepted rather than requiring
 * the picker's placeholder option. A model without capability data keeps the
 * placeholder check.
 */
export function reasoningEffortAllowedForModel(
  model: ClientModel,
  effort: ReasoningEffort,
): boolean {
  return (
    model.capabilities?.reasoning.efforts.length === 0 ||
    effortOptionsForModel(model).includes(effort)
  );
}
