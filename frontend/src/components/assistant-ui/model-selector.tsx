"use client";

import {
  memo,
  useState,
  useEffect,
  createContext,
  useContext,
  type ReactNode,
} from "react";
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

export type ModelOption = {
  id: string;
  name: string;
  description?: string;
  icon?: ReactNode;
  disabled?: boolean;
};

type ModelSelectorContextValue = {
  models: ModelOption[];
  value: string;
  onValueChange: (v: string) => void;
  setOpen: (o: boolean) => void;
};

const ModelSelectorContext = createContext<ModelSelectorContextValue | null>(null);

function useModelSelectorContext() {
  const ctx = useContext(ModelSelectorContext);
  if (!ctx) throw new Error("ModelSelector sub-components must be used within ModelSelector.Root");
  return ctx;
}

// ─── Root ──────────────────────────────────────────────────────────────────

export type ModelSelectorRootProps = {
  models: ModelOption[];
  value: string;
  onValueChange: (v: string) => void;
  open: boolean;
  onOpenChange: (o: boolean) => void;
  children: ReactNode;
};

function ModelSelectorRoot({ models, value, onValueChange, open, onOpenChange, children }: ModelSelectorRootProps) {
  return (
    <ModelSelectorContext.Provider value={{ models, value, onValueChange, setOpen: onOpenChange }}>
      <PickerRoot open={open} onOpenChange={onOpenChange}>
        {children}
      </PickerRoot>
    </ModelSelectorContext.Provider>
  );
}

// ─── Trigger ───────────────────────────────────────────────────────────────

export type ModelSelectorTriggerProps = VariantProps<typeof pickerTriggerVariants> & {
  className?: string;
};

function ModelSelectorTrigger({ className, variant = "ghost", size = "sm" }: ModelSelectorTriggerProps) {
  const { models, value } = useModelSelectorContext();
  const current = models.find((m) => m.id === value) ?? models[0];
  return (
    <PickerTrigger
      variant={variant}
      size={size}
      className={cn("aui-model-selector-trigger text-muted-foreground", className)}
      aria-label="Select model"
    >
      <span className="font-medium">{current?.name ?? value}</span>
    </PickerTrigger>
  );
}

// ─── Content ───────────────────────────────────────────────────────────────

function ModelSelectorContent({ className }: { className?: string }) {
  const { models, value, onValueChange, setOpen } = useModelSelectorContext();
  return (
    <PickerContent className={cn("min-w-44", className)} align="end">
      {models.map((model) => (
        <PickerItem
          key={model.id}
          selected={model.id === value}
          description={model.description}
          disabled={model.disabled}
          onClick={() => { onValueChange(model.id); setOpen(false); }}
        >
          {model.icon ? (
            <span className="flex items-center gap-2">
              <span className="flex size-4 shrink-0 items-center justify-center [&_svg]:size-4">
                {model.icon}
              </span>
              {model.name}
            </span>
          ) : model.name}
        </PickerItem>
      ))}
    </PickerContent>
  );
}

// ─── Composed ModelSelector ────────────────────────────────────────────────

export type ModelSelectorProps = {
  models: ModelOption[];
  value?: string;
  onValueChange?: (v: string) => void;
  defaultValue?: string;
  variant?: VariantProps<typeof pickerTriggerVariants>["variant"];
  size?: VariantProps<typeof pickerTriggerVariants>["size"];
  contentClassName?: string;
};

const ModelSelectorImpl = ({
  value: controlledValue,
  onValueChange: controlledOnValueChange,
  defaultValue,
  models,
  variant,
  size,
  contentClassName,
}: ModelSelectorProps) => {
  const isControlled = controlledValue !== undefined;
  const [internalValue, setInternalValue] = useState(
    () => defaultValue ?? models[0]?.id ?? "",
  );
  const [open, setOpen] = useState(false);

  const value = isControlled ? controlledValue! : internalValue;
  const onValueChange = controlledOnValueChange ?? setInternalValue;

  const api = useAssistantApi();
  useEffect(() => {
    const config = { config: { modelName: value } };
    return api.modelContext().register({ getModelContext: () => config });
  }, [api, value]);

  return (
    <ModelSelectorRoot
      models={models}
      value={value}
      onValueChange={onValueChange}
      open={open}
      onOpenChange={setOpen}
    >
      <ModelSelectorTrigger variant={variant} size={size} />
      <ModelSelectorContent className={contentClassName} />
    </ModelSelectorRoot>
  );
};

type ModelSelectorComponent = typeof ModelSelectorImpl & {
  displayName?: string;
  Root: typeof ModelSelectorRoot;
  Trigger: typeof ModelSelectorTrigger;
  Content: typeof ModelSelectorContent;
};

const ModelSelector = memo(ModelSelectorImpl) as unknown as ModelSelectorComponent;
ModelSelector.displayName = "ModelSelector";
ModelSelector.Root = ModelSelectorRoot;
ModelSelector.Trigger = ModelSelectorTrigger;
ModelSelector.Content = ModelSelectorContent;

export { ModelSelector, ModelSelectorRoot, ModelSelectorTrigger, ModelSelectorContent };
