"use client";

import React, { createContext, useCallback, useContext, useSyncExternalStore } from "react";

export type BackgroundTheme = "stars" | "meteors" | "rain" | "mesh" | "starfield" | "particles" | "darkveil";

const STORAGE_KEY = "dh-background-theme";
const THEME_CHANGE_EVENT = "dh-theme-change";
const THEMES: BackgroundTheme[] = [
  "stars",
  "meteors",
  "rain",
  "mesh",
  "starfield",
  "particles",
  "darkveil",
];

function parseTheme(raw: string | null): BackgroundTheme {
  if (!raw || raw === "none") return "stars";
  return THEMES.includes(raw as BackgroundTheme) ? (raw as BackgroundTheme) : "stars";
}

function getThemeSnapshot(): BackgroundTheme {
  if (typeof window === "undefined") return "stars";
  try {
    return parseTheme(window.localStorage.getItem(STORAGE_KEY));
  } catch {
    return "stars";
  }
}

function subscribeTheme(callback: () => void): () => void {
  if (typeof window === "undefined") {
    return () => {};
  }

  const onThemeChange = () => callback();
  window.addEventListener("storage", onThemeChange);
  window.addEventListener(THEME_CHANGE_EVENT, onThemeChange);
  return () => {
    window.removeEventListener("storage", onThemeChange);
    window.removeEventListener(THEME_CHANGE_EVENT, onThemeChange);
  };
}

interface ThemeContextValue {
  theme: BackgroundTheme;
  setTheme: (t: BackgroundTheme) => void;
}

const ThemeContext = createContext<ThemeContextValue>({
  theme: "stars",
  setTheme: () => {},
});

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const theme = useSyncExternalStore<BackgroundTheme>(
    subscribeTheme,
    getThemeSnapshot,
    () => "stars",
  );

  const setTheme = useCallback((t: BackgroundTheme) => {
    try {
      window.localStorage.setItem(STORAGE_KEY, t);
    } catch {
      return;
    }
    if (typeof window !== "undefined") {
      window.dispatchEvent(new Event(THEME_CHANGE_EVENT));
    }
  }, []);

  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  return useContext(ThemeContext);
}
