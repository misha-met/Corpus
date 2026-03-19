"use client";

import { ChevronRight } from "lucide-react";
import { SourcePanel } from "@/components/source-panel";
import { CHROME_TRANSITION, type ChromeStyles } from "@/lib/theme-constants";

interface SourcePanelContainerProps {
  chatMode: "rag" | "freeform";
  isPanelCollapsed: boolean;
  chromeStyles: ChromeStyles;
  selectedSourceIds: string[];
  onSelectedSourceIdsChange: (ids: string[]) => void;
  onCollapse: () => void;
  onExpand: () => void;
  onSourcesChanged: () => void;
}

export function SourcePanelContainer({
  chatMode,
  isPanelCollapsed,
  chromeStyles,
  selectedSourceIds,
  onSelectedSourceIdsChange,
  onCollapse,
  onExpand,
  onSourcesChanged,
}: SourcePanelContainerProps) {
  const hidden = chatMode !== "rag";

  return (
    <aside
      className="relative shrink-0 overflow-hidden flex flex-col"
      aria-label="Source documents"
      style={{
        width: hidden
          ? "0"
          : isPanelCollapsed
            ? "2.5rem"
            : "min(30%, 24rem)",
        minWidth: hidden
          ? "0"
          : isPanelCollapsed
            ? "2.5rem"
            : "14rem",
        opacity: hidden ? 0 : 1,
        pointerEvents: hidden ? "none" : "auto",
        transition: `width 280ms cubic-bezier(0.4,0,0.2,1), min-width 280ms cubic-bezier(0.4,0,0.2,1), opacity 240ms ease, ${CHROME_TRANSITION}`,
        background: chromeStyles.bg,
        borderRight: `1px solid ${chromeStyles.borderColor}`,
        boxShadow:
          "2px 0 8px rgba(0,0,0,0.45), 1px 0 2px rgba(0,0,0,0.35), inset -1px 0 0 rgba(255,255,255,0.04)",
        backdropFilter: chromeStyles.backdrop,
        WebkitBackdropFilter: chromeStyles.backdrop,
        willChange: "opacity, background, backdrop-filter",
        isolation: "isolate",
      }}
    >
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "linear-gradient(180deg, rgba(0,0,0,0.62) 0%, rgba(0,0,0,0.68) 100%)",
          opacity: chromeStyles.fadeOpacity,
          transition:
            "opacity 620ms cubic-bezier(0.22,1,0.36,1)",
        }}
      />

      <div className="relative z-10 flex h-full flex-col">
        {isPanelCollapsed ? (
          <div className="flex flex-col items-center pt-3">
            <button
              onClick={onExpand}
              className="p-1.5 text-muted-foreground hover:text-foreground hover:bg-white/5 rounded transition-colors"
              title="Expand sources panel"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        ) : (
          <SourcePanel
            selectedSourceIds={selectedSourceIds}
            onSelectedSourceIdsChange={onSelectedSourceIdsChange}
            onCollapse={onCollapse}
            onSourcesChanged={onSourcesChanged}
          />
        )}
      </div>
    </aside>
  );
}

