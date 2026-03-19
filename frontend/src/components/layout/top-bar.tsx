"use client";

import { useState } from "react";
import {
  Map as MapIcon,
  PaletteIcon,
  Plus,
  UserRound,
  FileText,
  MessageSquare,
  Clock,
} from "lucide-react";
import { PickerRoot, PickerTrigger, PickerContent, PickerItem } from "@/components/ui/picker";
import {
  THEMES,
  type ChromeStyles,
  CLS_TAB_ACTIVE,
  CLS_TAB_INACTIVE,
  CLS_TAB_BASE,
  CHROME_TRANSITION,
} from "@/lib/theme-constants";
import type { BackgroundTheme } from "@/context/theme-context";

/* ── small internal button ────────────────────────────────────────────────── */

function TabButton({
  active,
  onClick,
  icon,
  label,
  title,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
  title?: string;
}) {
  return (
    <button
      onClick={onClick}
      className={`${CLS_TAB_BASE} ${active ? CLS_TAB_ACTIVE : CLS_TAB_INACTIVE}`}
      title={title ?? label}
      aria-label={title ?? label}
      aria-pressed={active}
    >
      {icon}
      {label}
    </button>
  );
}

/* ── TopBar ───────────────────────────────────────────────────────────────── */

interface TopBarProps {
  chatMode: "rag" | "freeform";
  isMapOpen: boolean;
  isPeopleOpen: boolean;
  showHistory: boolean;
  theme: BackgroundTheme;
  chromeStyles: ChromeStyles;
  onSetMode: (mode: "rag" | "freeform") => void;
  onToggleMap: () => void;
  onTogglePeople: () => void;
  onNewChat: () => void;
  onSetTheme: (theme: BackgroundTheme) => void;
  onToggleHistory: () => void;
}

export function TopBar({
  chatMode,
  isMapOpen,
  isPeopleOpen,
  showHistory,
  theme,
  chromeStyles,
  onSetMode,
  onToggleMap,
  onTogglePeople,
  onNewChat,
  onSetTheme,
  onToggleHistory,
}: TopBarProps) {
  const [themeOpen, setThemeOpen] = useState(false);
  const iconCls = "w-3.5 h-3.5 shrink-0";

  return (
    <header
      className="relative z-30 flex items-center gap-2 px-4 py-2 justify-between w-full shrink-0"
      style={{
        background: chromeStyles.bg,
        borderBottom: `1px solid ${chromeStyles.borderColor}`,
        boxShadow:
          "0 2px 8px rgba(0,0,0,0.45), 0 1px 2px rgba(0,0,0,0.35), inset 0 -1px 0 rgba(255,255,255,0.04)",
        backdropFilter: chromeStyles.backdrop,
        WebkitBackdropFilter: chromeStyles.backdrop,
        transition: CHROME_TRANSITION,
        isolation: "isolate",
      }}
    >
      {/* Overlay fade */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "linear-gradient(180deg, rgba(0,0,0,0.58) 0%, rgba(0,0,0,0.72) 100%)",
          opacity: chromeStyles.fadeOpacity,
          transition: "opacity 560ms cubic-bezier(0.22,1,0.36,1)",
        }}
      />

      {/* Left — mode tabs + overlay toggles */}
      <nav className="relative z-10 flex items-center gap-2" aria-label="Mode tabs">
        <TabButton
          active={chatMode === "rag"}
          onClick={() => onSetMode("rag")}
          icon={<FileText className={iconCls} />}
          label="RAG Mode"
        />
        <TabButton
          active={chatMode === "freeform"}
          onClick={() => onSetMode("freeform")}
          icon={<MessageSquare className={iconCls} />}
          label="Non-RAG Mode"
        />
        <TabButton
          active={isMapOpen}
          onClick={onToggleMap}
          icon={<MapIcon className={iconCls} />}
          label="Map"
        />
        <TabButton
          active={isPeopleOpen}
          onClick={onTogglePeople}
          icon={<UserRound className={iconCls} />}
          label="People"
        />
      </nav>

      {/* Right — actions */}
      <div className="relative z-10 flex items-center gap-2">
        <button
          onClick={onNewChat}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-colors text-white/40 hover:text-white/70 hover:bg-white/8"
          title="New chat"
        >
          <Plus className={iconCls} />
          New Chat
        </button>

        <PickerRoot open={themeOpen} onOpenChange={setThemeOpen}>
          <PickerTrigger
            variant="ghost"
            size="sm"
            className="text-white/40 hover:text-white/70 hover:bg-white/8 data-[state=open]:bg-white/15 data-[state=open]:text-white data-[state=open]:border data-[state=open]:border-white/25"
            title="Background theme"
          >
            <PaletteIcon className={iconCls} />
            Theme
          </PickerTrigger>
          <PickerContent align="end" className="min-w-36">
            {THEMES.map((t) => (
              <PickerItem
                key={t.id}
                selected={theme === t.id}
                onClick={() => {
                  onSetTheme(t.id);
                  setThemeOpen(false);
                }}
              >
                {t.label}
              </PickerItem>
            ))}
          </PickerContent>
        </PickerRoot>

        <TabButton
          active={showHistory}
          onClick={onToggleHistory}
          icon={<Clock className={iconCls} />}
          label="Chat History"
        />
      </div>
    </header>
  );
}

