"use client";

import { memo, useEffect, useState } from "react";
import { type VariantProps } from "class-variance-authority";
import { useAssistantApi } from "@assistant-ui/react";
import { cn } from "@/lib/utils";
import {
  PickerRoot,
  PickerTrigger,
  PickerContent,
  PickerItem,
  PickerSeparator,
  pickerTriggerVariants,
} from "@/components/ui/picker";
import { useAppDispatch, useAppState } from "@/context/app-context";

// ---------------------------------------------------------------------------
// Intent options
// ---------------------------------------------------------------------------

export interface IntentOption {
  id: string;
  name: string;
  description: string;
}

export const INTENT_OPTIONS: IntentOption[] = [
  {
    id: "auto",
    name: "Auto",
    description: "The system automatically chooses the best response mode for your question.",
  },
  {
    id: "summarise",
    name: "Summarise",
    description: "Pulls out the key points and main ideas concisely.",
  },
  {
    id: "explain",
    name: "Explain",
    description: "Breaks down complex or technical content into plain language.",
  },
  {
    id: "analyze",
    name: "Analyze",
    description: "Examines deeper meanings, themes, and significance in the text.",
  },
  {
    id: "compare",
    name: "Compare",
    description: "Shows similarities and differences between multiple ideas or documents.",
  },
  {
    id: "critique",
    name: "Critique",
    description: "Highlights the strengths, weaknesses, and limitations in the text.",
  },
  {
    id: "factual",
    name: "Factual",
    description: "Provides a direct, specific answer to a concrete question.",
  },
  {
    id: "collection",
    name: "Collection",
    description: "Describes the overall themes and scope of your full document set.",
  },
  {
    id: "extract",
    name: "Extract",
    description: "Pulls out specific data like names, dates, or figures into a list.",
  },
  {
    id: "timeline",
    name: "Timeline",
    description: "Organizes events in chronological order.",
  },
  {
    id: "how_to",
    name: "How-To",
    description: "Presents clear, step-by-step instructions from the document.",
  },
  {
    id: "quote_evidence",
    name: "Quote / Evidence",
    description: "Returns relevant direct quotes to support a claim or question.",
  },
];

// ---------------------------------------------------------------------------
// IntentSelector
// ---------------------------------------------------------------------------

export type IntentSelectorProps = VariantProps<typeof pickerTriggerVariants> & {
  contentClassName?: string;
};

const IntentSelectorImpl = ({
  variant = "ghost",
  size = "sm",
  contentClassName,
}: IntentSelectorProps) => {
  const dispatch = useAppDispatch();
  const { intentOverride } = useAppState();
  const api = useAssistantApi();
  const [open, setOpen] = useState(false);

  useEffect(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const config = { config: { intentOverride } as any };
    return api.modelContext().register({ getModelContext: () => config });
  }, [api, intentOverride]);

  const handleSelect = (id: string) => {
    dispatch({ type: "SET_INTENT_OVERRIDE", intentOverride: id });
    setOpen(false);
  };

  const currentOption = INTENT_OPTIONS.find((o) => o.id === intentOverride) ?? INTENT_OPTIONS[0];
  const isAuto = intentOverride === "auto";

  return (
    <PickerRoot open={open} onOpenChange={setOpen}>
      <PickerTrigger
        variant={variant}
        size={size}
        className={cn(
          "aui-intent-selector-trigger text-muted-foreground gap-1.5",
          !isAuto && "text-blue-400",
        )}
        aria-label="Select response mode"
      >
        {isAuto ? (
          <span className="font-medium">Auto</span>
        ) : (
          <span className="flex items-center gap-1.5">
            <span className="size-1.5 rounded-full bg-blue-400 shrink-0" />
            <span className="font-medium">{currentOption.name}</span>
          </span>
        )}
      </PickerTrigger>

      <PickerContent
        className={cn("w-64 max-h-72 overflow-y-auto", contentClassName)}
        align="end"
      >
        {/* Auto — always first */}
        <PickerItem
          selected={intentOverride === INTENT_OPTIONS[0].id}
          description={INTENT_OPTIONS[0].description}
          onClick={() => handleSelect(INTENT_OPTIONS[0].id)}
        >
          {INTENT_OPTIONS[0].name}
        </PickerItem>
        <PickerSeparator />
        {INTENT_OPTIONS.slice(1).map((option) => (
          <PickerItem
            key={option.id}
            selected={intentOverride === option.id}
            description={option.description}
            onClick={() => handleSelect(option.id)}
          >
            {option.name}
          </PickerItem>
        ))}
      </PickerContent>
    </PickerRoot>
  );
};

export const IntentSelector = memo(IntentSelectorImpl);
IntentSelector.displayName = "IntentSelector";
