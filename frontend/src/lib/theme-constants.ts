import type { BackgroundTheme } from "@/context/theme-context";

/** Glass-panel style tokens per background theme. */
export const GLASS: Record<
  BackgroundTheme,
  { bg: string; backdrop: string; border: string }
> = {
  meteors: {
    bg: "rgba(255,255,255,0.0)",
    backdrop: "blur(7px)",
    border: "rgba(255,255,255,0.060)",
  },
  rain: {
    bg: "rgba(0,0,0,0.175)",
    backdrop: "blur(9px) saturate(100%)",
    border: "rgba(255,255,255,0.000)",
  },
  mesh: {
    bg: "rgba(0,0,0,0.055)",
    backdrop: "blur(9px) saturate(110%)",
    border: "rgba(255,255,255,0.000)",
  },
  starfield: {
    bg: "rgba(0,0,0,0.165)",
    backdrop: "blur(10px) saturate(100%)",
    border: "rgba(255,255,255,0.000)",
  },
  particles: {
    bg: "rgba(255,255,255,0.030)",
    backdrop: "blur(4px)",
    border: "rgba(255,255,255,0.12)",
  },
  stars: {
    bg: "rgba(0,0,0,0.145)",
    backdrop: "blur(4px) saturate(100%)",
    border: "rgba(255,255,255,0.000)",
  },
  darkveil: {
    bg: "rgba(0,0,0,0.250)",
    backdrop: "blur(6px) saturate(110%)",
    border: "rgba(255,255,255,0.000)",
  },
};

/** Themes that need a solid dark page background instead of var(--background). */
export const DARK_BG_THEMES = new Set<BackgroundTheme>([
  "meteors",
  "stars",
  "starfield",
  "darkveil",
]);

/** Ordered list for the theme-picker dropdown. */
export const THEMES: { id: BackgroundTheme; label: string }[] = [
  { id: "stars", label: "Stars" },
  { id: "meteors", label: "Meteors" },
  { id: "rain", label: "Rain" },
  { id: "mesh", label: "Gradient" },
  { id: "starfield", label: "Starfield" },
  { id: "particles", label: "Particles" },
  { id: "darkveil", label: "Dark Veil" },
];

/** Computed chrome styles used by header, source panel, and overlay scrim. */
export interface ChromeStyles {
  bg: string;
  backdrop: string;
  borderColor: string;
  fadeOpacity: number;
}

export const CLS_TAB_ACTIVE = "bg-white/15 text-white border border-white/25 shadow-sm";
export const CLS_TAB_INACTIVE = "text-white/40 hover:text-white/70 hover:bg-white/8";
export const CLS_TAB_BASE = "flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all";
export const OVERLAY_TRANSITION = "opacity 420ms cubic-bezier(0.22,1,0.36,1), transform 520ms cubic-bezier(0.22,1,0.36,1), filter 420ms ease";
export const CHROME_TRANSITION = "background 520ms cubic-bezier(0.22,1,0.36,1), border-color 520ms cubic-bezier(0.22,1,0.36,1), backdrop-filter 520ms cubic-bezier(0.22,1,0.36,1)";
