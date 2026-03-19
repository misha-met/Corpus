"use client";

import { useState, useEffect, useCallback } from "react";

/**
 * Like useState<number>, but hydrates from localStorage on mount and
 * persists every change back.  A clamp function is applied on both
 * read and write so the value is always within bounds.
 */
export function usePersistedState(
  key: string,
  defaultValue: number,
  clampFn: (v: number) => number,
): [number, (v: number) => void] {
  const [value, setValue] = useState(defaultValue);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(key);
      if (raw) setValue(clampFn(Number(raw)));
    } catch { /* storage unavailable */ }
    setHydrated(true);
  }, [key, clampFn]);

  useEffect(() => {
    if (!hydrated) return;
    try { window.localStorage.setItem(key, String(value)); }
    catch { /* storage full or unavailable */ }
  }, [key, value, hydrated]);

  const set = useCallback(
    (v: number) => setValue(clampFn(v)),
    [clampFn],
  );

  return [value, set];
}
