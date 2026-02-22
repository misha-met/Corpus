/**
 * citation-meta-store.ts
 *
 * Persists user-provided citation reference strings in localStorage,
 * keyed by source_id.
 *
 * A citation reference is a freeform string the user types at ingest time,
 * e.g. "Smith, J. et al. (2024) 'Climate Change and Food Security'"
 *
 * This is the single source of truth for the copy-as-citation feature.
 * If no reference was saved for a source, the formatter falls back to the
 * source_id (filename stem) and page numbers.
 */

const STORAGE_KEY = "dh-citation-meta-v1";

type CitationMetaStore = Record<string, string>;

function loadStore(): CitationMetaStore {
  if (typeof window === "undefined") return {};
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as CitationMetaStore) : {};
  } catch {
    return {};
  }
}

function saveStore(store: CitationMetaStore): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
  } catch {
    // localStorage quota exceeded — silently skip
  }
}

/** Save a citation reference string for a source. */
export function saveCitationMeta(sourceId: string, citationRef: string): void {
  const ref = citationRef.trim();
  const store = loadStore();
  if (ref) {
    store[sourceId] = ref;
  } else {
    delete store[sourceId];
  }
  saveStore(store);
}

/** Retrieve the stored citation reference for a source, or null. */
export function getCitationMeta(sourceId: string): string | null {
  const store = loadStore();
  return store[sourceId] ?? null;
}

/** Remove citation metadata for a source (call on source deletion). */
export function deleteCitationMeta(sourceId: string): void {
  const store = loadStore();
  delete store[sourceId];
  saveStore(store);
}
