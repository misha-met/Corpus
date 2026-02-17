/**
 * event-parser.ts — Phase 3
 *
 * Typed parser for the custom data parts emitted by the DH Notebook backend
 * over the AI SDK v6 UI message stream.
 *
 * Wire format (produced by src/stream_protocol.py):
 *   data: {"type": "data-<name>", "data": <inner payload>}
 *
 * On the client the AI SDK v6 `useChat` `onData` callback receives objects
 * with this shape, so `dataPart.type` is e.g. "data-status" and
 * `dataPart.data` is the inner payload object.
 *
 * This module is pure — no React, no side-effects. Import types freely.
 */

// ---------------------------------------------------------------------------
// Citation type (matches CitationListEvent in src/query_events.py)
// ---------------------------------------------------------------------------

export interface Citation {
  /** 1-based citation number matching [N] markers in the response text.
   * Mapped from the backend field "index" (rag_engine.py emits { index: N, ... }). */
  number: number;
  chunk_id: string;
  source_id: string;
  /** Mapped from backend field "page_number". */
  page?: number | null;
  header_path?: string | null;
  /** Raw chunk text from the retrieval result. */
  chunk_text?: string;
  /** Optional verified highlight anchor computed post-generation. */
  highlight_text?: string;
}

// ---------------------------------------------------------------------------
// Discriminated union of all custom event shapes
// ---------------------------------------------------------------------------

export interface StatusEvent {
  type: "status";
  status: string;
}

export interface IntentEvent {
  type: "intent";
  intent: string;
  confidence: number;
  method: string;
}

export interface SourcesEvent {
  type: "sources";
  /** Source IDs returned by retrieval (matches sourceIds wire field) */
  sourceIds: string[];
}

export interface CitationsEvent {
  type: "citations";
  citations: Citation[];
}

export interface ErrorEvent {
  type: "error";
  error: {
    code: string;
    message: string;
    metadata?: Record<string, unknown>;
  };
}

export interface FinishStepEvent {
  type: "finish-step";
  finishReason: string;
  isContinued: boolean;
}

/** All custom event variants that the backend can produce */
export type CustomEvent =
  | StatusEvent
  | IntentEvent
  | SourcesEvent
  | CitationsEvent
  | ErrorEvent
  | FinishStepEvent;

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/**
 * Parse a raw AI SDK v6 `onData` data-part into a typed `CustomEvent`.
 *
 * Receives the full dataPart object `{ type: string, data: unknown }` that
 * the `onData` callback forwards from `useChat`. Returns `null` for any
 * unrecognised or malformed payload so callers can safely ignore it.
 *
 * @example
 * ```ts
 * useChat({
 *   onData: (dataPart) => {
 *     const event = parseCustomEvent(dataPart);
 *     if (event) dispatch(event);
 *   },
 * });
 * ```
 */
export function parseCustomEvent(dataPart: unknown): CustomEvent | null {
  if (!dataPart || typeof dataPart !== "object") return null;

  const part = dataPart as Record<string, unknown>;
  const partType = part["type"];

  if (typeof partType !== "string") return null;

  switch (partType) {
    case "data-status": {
      const d = part["data"] as Record<string, unknown> | null | undefined;
      if (typeof d?.["status"] === "string") {
        return { type: "status", status: d["status"] };
      }
      return null;
    }

    case "data-intent": {
      const d = part["data"] as Record<string, unknown> | null | undefined;
      if (
        typeof d?.["intent"] === "string" &&
        typeof d?.["confidence"] === "number" &&
        typeof d?.["method"] === "string"
      ) {
        return {
          type: "intent",
          intent: d["intent"],
          confidence: d["confidence"],
          method: d["method"],
        };
      }
      return null;
    }

    case "data-sources": {
      const d = part["data"] as Record<string, unknown> | null | undefined;
      const ids = d?.["sourceIds"];
      if (Array.isArray(ids)) {
        return { type: "sources", sourceIds: ids as string[] };
      }
      return null;
    }

    case "data-citations": {
      const d = part["data"] as Record<string, unknown> | null | undefined;
      const cits = d?.["citations"];
      if (Array.isArray(cits)) {
        // Remap backend field names to the Citation interface:
        //   "index"       → number   (backend uses "index" for the 1-based citation number)
        //   "page_number" → page     (backend field is "page_number", interface field is "page")
        const citations: Citation[] = (cits as Record<string, unknown>[]).map((c) => ({
          number: (c["index"] as number) ?? 0,
          chunk_id: (c["chunk_id"] as string) ?? "",
          source_id: (c["source_id"] as string) ?? "",
          page: (c["page_number"] as number | null) ?? null,
          header_path: (c["header_path"] as string | null) ?? null,
          chunk_text: (c["chunk_text"] as string | undefined) ?? undefined,
          highlight_text: (c["highlight_text"] as string | undefined) ?? undefined,
        }));
        return { type: "citations", citations };
      }
      return null;
    }

    case "data-error": {
      const d = part["data"] as Record<string, unknown> | null | undefined;
      const err = d?.["error"] as Record<string, unknown> | null | undefined;
      if (typeof err?.["code"] === "string" && typeof err?.["message"] === "string") {
        const event: ErrorEvent = {
          type: "error",
          error: { code: err["code"], message: err["message"] },
        };
        if (err["metadata"] && typeof err["metadata"] === "object") {
          event.error.metadata = err["metadata"] as Record<string, unknown>;
        }
        return event;
      }
      return null;
    }

    case "data-finish-step": {
      const d = part["data"] as Record<string, unknown> | null | undefined;
      return {
        type: "finish-step",
        finishReason: typeof d?.["finishReason"] === "string" ? d["finishReason"] : "stop",
        isContinued: Boolean(d?.["isContinued"]),
      };
    }

    default:
      return null;
  }
}
