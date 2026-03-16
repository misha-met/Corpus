/**
 * session-store.ts — IndexedDB wrapper for chat session persistence.
 *
 * Stores ChatSession objects (both RAG and freeform) keyed by session ID.
 * Uses `put()` so re-saving the same ID replaces the existing record —
 * this works with the stable sessionIdRef pattern in FreeformChatPanel.
 */

const DB_NAME = "dh-notebook-sessions";
const DB_VERSION = 2;   // bumped from 1 → 2 to force onupgradeneeded on existing DBs
const STORE_NAME = "sessions";

export interface FreeChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  /** Captured <think>…</think> reasoning chain, set when thinking mode is on */
  thinkingContent?: string;
  timestamp: number;
  traceId?: string;
  spanId?: string;
}

export interface ChatSession {
  id: string;
  /** "rag" for document-QA sessions, "freeform" for non-RAG sessions */
  mode: "rag" | "freeform";
  /** Display title — first user message, truncated */
  title: string;
  /** Full message history (freeform) or empty for RAG sessions */
  messages: FreeChatMessage[];
  createdAt: number;
  updatedAt: number;
}

// ---------------------------------------------------------------------------
// DB open helper (lazily cached)
// ---------------------------------------------------------------------------

let _dbPromise: Promise<IDBDatabase> | null = null;

function openDb(): Promise<IDBDatabase> {
  if (_dbPromise) return _dbPromise;
  _dbPromise = new Promise<IDBDatabase>((resolve, reject) => {
    if (typeof indexedDB === "undefined") {
      reject(new Error("IndexedDB not available"));
      return;
    }
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = (e) => {
      const db = (e.target as IDBOpenDBRequest).result;
      // Create the store on fresh install OR on upgrade from v1 (which may
      // have been opened without the store if the old code ran first).
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: "id" });
        store.createIndex("updatedAt", "updatedAt");
        store.createIndex("mode", "mode");
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => {
      _dbPromise = null; // allow retry on next call
      reject(req.error);
    };
    req.onblocked = () => {
      _dbPromise = null;
      reject(new Error("IDBDatabase blocked — please close other tabs"));
    };
  });
  return _dbPromise;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** Save (upsert) a session — replaces existing record if same `id`. */
export async function saveSession(session: ChatSession): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const req = tx.objectStore(STORE_NAME).put(session);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

/** Load a single session by ID, or null if not found. */
export async function loadSession(id: string): Promise<ChatSession | null> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const req = tx.objectStore(STORE_NAME).get(id);
    req.onsuccess = () => resolve((req.result as ChatSession) ?? null);
    req.onerror = () => reject(req.error);
  });
}

/** Return all sessions, sorted newest first. */
export async function listSessions(): Promise<ChatSession[]> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const req = tx.objectStore(STORE_NAME).getAll();
    req.onsuccess = () => {
      const sessions = (req.result as ChatSession[]) ?? [];
      sessions.sort((a, b) => b.updatedAt - a.updatedAt);
      resolve(sessions);
    };
    req.onerror = () => reject(req.error);
  });
}

/** Delete a session by ID. */
export async function deleteSession(id: string): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const req = tx.objectStore(STORE_NAME).delete(id);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

/** Derive a display title from the first user message. */
export function deriveTitle(messages: FreeChatMessage[]): string {
  const first = messages.find((m) => m.role === "user");
  if (!first) return "Untitled session";
  return first.content.length > 60
    ? first.content.slice(0, 57) + "..."
    : first.content;
}
