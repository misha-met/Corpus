"use client"

/**
 * PickerDropdown — Popover-based replacement for Radix Select in inline pickers.
 *
 * Unlike Radix Select (which uses a shared DismissableLayer that forces
 * exactly one open at a time), Popover has fully independent open state per
 * instance, so two pickers in the same toolbar can both be open at once.
 */

import * as React from "react"
import * as PopoverPrimitive from "@radix-ui/react-popover"
import { CheckIcon, ChevronDownIcon } from "lucide-react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

// ─── Root / Trigger pass-throughs ────────────────────────────────────────────

export const PickerRoot = PopoverPrimitive.Root
export const PickerPortal = PopoverPrimitive.Portal
export const PickerAnchor = PopoverPrimitive.Anchor

// ─── Trigger button ───────────────────────────────────────────────────────────

export const pickerTriggerVariants = cva(
  "flex items-center justify-between gap-1.5 rounded-md border text-sm whitespace-nowrap transition-colors outline-none disabled:cursor-not-allowed disabled:opacity-50",
  {
    variants: {
      variant: {
        outline: "border-input bg-background hover:bg-accent hover:text-accent-foreground",
        ghost:   "border-transparent bg-transparent hover:bg-accent hover:text-accent-foreground",
        muted:   "border-transparent bg-secondary hover:bg-secondary/80",
      },
      size: {
        sm:      "h-8 px-2 text-xs",
        default: "h-9 px-3 py-2",
        lg:      "h-10 px-4",
      },
    },
    defaultVariants: {
      variant: "ghost",
      size: "sm",
    },
  },
)

export interface PickerTriggerProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof pickerTriggerVariants> {
  showChevron?: boolean
}

export const PickerTrigger = React.forwardRef<HTMLButtonElement, PickerTriggerProps>(
  ({ className, variant, size, showChevron = true, children, ...props }, ref) => (
    <PopoverPrimitive.Trigger asChild>
      <button
        ref={ref}
        type="button"
        className={cn(pickerTriggerVariants({ variant, size }), className)}
        {...props}
      >
        {children}
        {showChevron && <ChevronDownIcon className="size-3 opacity-50 shrink-0" />}
      </button>
    </PopoverPrimitive.Trigger>
  ),
)
PickerTrigger.displayName = "PickerTrigger"

// ─── Dropdown content panel ───────────────────────────────────────────────────

export interface PickerContentProps
  extends React.ComponentPropsWithoutRef<typeof PopoverPrimitive.Content> {}

export const PickerContent = React.forwardRef<
  React.ElementRef<typeof PopoverPrimitive.Content>,
  PickerContentProps
>(({ className, sideOffset = 6, style, children, ...props }, ref) => (
  <PopoverPrimitive.Portal>
    <PopoverPrimitive.Content
      ref={ref}
      sideOffset={sideOffset}
      className={cn(
        "z-50 min-w-32 overflow-hidden rounded-xl p-1",
        "backdrop-blur-xl text-popover-foreground",
        "data-[state=open]:animate-in data-[state=closed]:animate-out",
        "data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0",
        "data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95",
        "data-[side=bottom]:slide-in-from-top-2 data-[side=top]:slide-in-from-bottom-2",
        "data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2",
        className,
      )}
      style={{
        background: "rgba(10,10,10,0.92)",
        border: "1px solid rgba(255,255,255,0.10)",
        boxShadow:
          "0 8px 32px rgba(0,0,0,0.7), 0 2px 8px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05)",
        ...style,
      }}
      {...props}
    >
      {children}
    </PopoverPrimitive.Content>
  </PopoverPrimitive.Portal>
))
PickerContent.displayName = "PickerContent"

// ─── Individual option row ────────────────────────────────────────────────────

export interface PickerItemProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  selected?: boolean
  /** Render description text below the label */
  description?: React.ReactNode
}

export const PickerItem = React.forwardRef<HTMLButtonElement, PickerItemProps>(
  ({ className, selected, description, children, ...props }, ref) => (
    <button
      ref={ref}
      type="button"
      role="option"
      aria-selected={selected}
      className={cn(
        "relative flex w-full cursor-default select-none items-start gap-2 rounded-lg py-2 pl-3 pr-9 text-sm outline-none text-left",
        "hover:bg-white/8 focus-visible:bg-white/8 transition-colors",
        "disabled:pointer-events-none disabled:opacity-50",
        className,
      )}
      {...props}
    >
      {/* Check indicator */}
      <span className="absolute right-3 top-2.5 flex size-4 shrink-0 items-center justify-center">
        {selected && <CheckIcon className="size-4" />}
      </span>
      <span className="flex flex-col gap-0.5 min-w-0">
        <span className="font-medium leading-tight truncate">{children}</span>
        {description && (
          <span className="text-muted-foreground text-xs leading-snug">{description}</span>
        )}
      </span>
    </button>
  ),
)
PickerItem.displayName = "PickerItem"

// ─── Separator ───────────────────────────────────────────────────────────────

export function PickerSeparator({ className }: { className?: string }) {
  return (
    <div
      className={cn("-mx-1 my-1 h-px", className)}
      style={{ background: "rgba(255,255,255,0.08)" }}
    />
  )
}
