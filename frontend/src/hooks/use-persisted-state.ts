"use client";

import { useCallback, useSyncExternalStore } from "react";

const LOCAL_STORAGE_EVENT = "dh-local-storage-change";

function _readPersistedNumber(
  key: string,
  defaultValue: number,
  clampFn: (v: number) => number,
): number {
  if (typeof window === "undefined") {
    return clampFn(defaultValue);
  }
  try {
    const raw = window.localStorage.getItem(key);
    if (raw !== null) {
      return clampFn(Number(raw));
    }
  } catch {
    /* storage unavailable */
  }
  return clampFn(defaultValue);
}

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
  const subscribe = useCallback(
    (onStoreChange: () => void) => {
      if (typeof window === "undefined") {
        return () => {};
      }

      const onStorage = (event: StorageEvent) => {
        if (!event.key || event.key === key) {
          onStoreChange();
        }
      };

      const onLocalStorageChange = (event: Event) => {
        const changedKey = (event as CustomEvent<{ key?: string }>).detail?.key;
        if (!changedKey || changedKey === key) {
          onStoreChange();
        }
      };

      window.addEventListener("storage", onStorage);
      window.addEventListener(LOCAL_STORAGE_EVENT, onLocalStorageChange);
      return () => {
        window.removeEventListener("storage", onStorage);
        window.removeEventListener(LOCAL_STORAGE_EVENT, onLocalStorageChange);
      };
    },
    [key],
  );

  const getSnapshot = useCallback(
    () => _readPersistedNumber(key, defaultValue, clampFn),
    [key, defaultValue, clampFn],
  );

  const getServerSnapshot = useCallback(
    () => clampFn(defaultValue),
    [defaultValue, clampFn],
  );

  const value = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  const set = useCallback(
    (v: number) => {
      if (typeof window === "undefined") {
        return;
      }

      const next = clampFn(v);
      try {
        window.localStorage.setItem(key, String(next));
      } catch {
        /* storage full or unavailable */
      }
      window.dispatchEvent(new CustomEvent(LOCAL_STORAGE_EVENT, { detail: { key } }));
    },
    [key, clampFn],
  );

  return [value, set];
}
