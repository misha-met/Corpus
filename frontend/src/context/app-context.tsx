/**
 * AppContext — custom application state for Corpus.
 *
 * What this context owns:
 *   - Per-query transient state: statusMessage, lastIntent, lastSources, citations
 *   - Persistent UI state: errorMessage, isLockBusy, activeCitation
 *
 * What this context deliberately does NOT own:
 *   - Busy / running state: read `useThread((t) => t.isRunning)` from
 *     @assistant-ui/react instead — it is the authoritative source of whether
 *     the assistant is currently generating a response.
 *   - Message history: owned and rendered entirely by the assistant-ui runtime.
 *   - Abort control: managed by the useChatRuntime / useThreadRuntime APIs.
 */
"use client";

import {
  createContext,
  useContext,
  useReducer,
  type Dispatch,
  type ReactNode,
} from "react";

import type { Citation } from "@/lib/event-parser";

// Re-export Citation so consumers can import it from this module if convenient
export type { Citation };

// ---------------------------------------------------------------------------
// Thinking step — one status update emitted during RAG pipeline execution
// ---------------------------------------------------------------------------

export interface ThinkingStep {
  /** Unique monotonic ID so React can key the list */
  id: number;
  /** User-friendly label for this pipeline step */
  message: string;
}

// ---------------------------------------------------------------------------
// State shape
// ---------------------------------------------------------------------------

export interface AppState {
  /** Human-readable status from stream annotations */
  statusMessage: string;
  /** Error message to display (cleared on next send) */
  errorMessage: string;
  /** Whether the error is a "server busy" (429) */
  isLockBusy: boolean;
  /** Intent from latest query */
  lastIntent: { intent: string; confidence: number; method: string } | null;
  /** Source IDs from latest query */
  lastSources: string[];
  /** Citations grouped by assistant message ID */
  citationsByMessage: Record<string, Citation[]>;
  /** The citation the user most recently clicked — read by CitationViewerModal */
  activeCitation: Citation | null;
  /** Ordered log of pipeline steps emitted during the current query (cleared on QUERY_STARTED) */
  thinkingSteps: ThinkingStep[];
  /** Trace ID and Span ID grouped by assistant message ID */
  traceInfoByMessage: Record<string, { traceId: string; spanId: string }>;
  /** Internal counter for generating unique ThinkingStep IDs */
  _stepCounter: number;
  /** The message ID of the assistant response currently generating */
  currentAssistantMessageId: string | null;
  /** User-selected intent override; "auto" means automatic classification */
  intentOverride: string;
  /** Active chat mode: "rag" for document-QA, "freeform" for non-RAG chat */
  chatMode: "rag" | "freeform";
}

const initialState: AppState = {
  statusMessage: "",
  errorMessage: "",
  isLockBusy: false,
  lastIntent: null,
  lastSources: [],
  citationsByMessage: {},
  activeCitation: null,
  thinkingSteps: [],
  traceInfoByMessage: {},
  _stepCounter: 0,
  currentAssistantMessageId: null,
  intentOverride: "auto",
  chatMode: "rag",
};

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

export type AppAction =
  /**
   * QUERY_STARTED — dispatched when a new query stream begins.
   * Resets all per-query state (statusMessage, lastIntent, lastSources,
   * citations, errorMessage, isLockBusy) so stale data from the previous
   * query is cleared before new stream events arrive.
   * Note: busy/running state is NOT tracked here — use
   * `useThread((t) => t.isRunning)` from @assistant-ui/react instead.
   */
  | { type: "QUERY_STARTED" }
  /**
   * QUERY_FINISHED — dispatched when the stream completes.
   * Clears the statusMessage left over from the last stream annotation.
   */
  | { type: "QUERY_FINISHED" }
  | { type: "SET_STATUS"; status: string }
  | { type: "SET_ERROR"; message: string; isLockBusy?: boolean }
  | { type: "CLEAR_ERROR" }
  | { type: "SET_INTENT"; intent: string; confidence: number; method: string }
  | { type: "SET_SOURCES"; sourceIds: string[] }
  | { type: "SET_CITATIONS"; citations: Citation[] }
  | { type: "SET_CURRENT_MESSAGE_ID"; messageId: string }
  | { type: "SET_ACTIVE_CITATION"; citation: Citation | null }
  | { type: "ADD_THINKING_STEP"; message: string }
  | { type: "SET_TRACE_INFO"; traceId: string; spanId: string }
  | { type: "SET_INTENT_OVERRIDE"; intentOverride: string }
  | { type: "SET_CHAT_MODE"; mode: "rag" | "freeform" };

// ---------------------------------------------------------------------------
// Reducer
// ---------------------------------------------------------------------------

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case "QUERY_STARTED":
      return {
        ...state,
        statusMessage: "Sending...",
        errorMessage: "",
        isLockBusy: false,
        lastIntent: null,
        lastSources: [],
        // Explicitly NOT clearing citationsByMessage so old messages retain their references drawer
        thinkingSteps: [],
        _stepCounter: 0,
      };
    case "QUERY_FINISHED":
      return {
        ...state,
        statusMessage: "",
      };

    case "SET_STATUS":
      return { ...state, statusMessage: action.status };
    case "SET_ERROR":
      return {
        ...state,
        statusMessage: "",
        errorMessage: action.message,
        isLockBusy: action.isLockBusy ?? false,
      };
    case "CLEAR_ERROR":
      return { ...state, errorMessage: "", isLockBusy: false };
    case "SET_INTENT":
      return {
        ...state,
        lastIntent: {
          intent: action.intent,
          confidence: action.confidence,
          method: action.method,
        },
      };
    case "SET_SOURCES":
      return { ...state, lastSources: action.sourceIds };
    case "SET_CITATIONS":
      if (!state.currentAssistantMessageId) {
        console.warn("Attempted to set citations but currentAssistantMessageId is null");
        return state;
      }
      return {
        ...state,
        citationsByMessage: {
          ...state.citationsByMessage,
          [state.currentAssistantMessageId]: action.citations
        }
      };
    case "SET_CURRENT_MESSAGE_ID":
      return { ...state, currentAssistantMessageId: action.messageId };
    case "SET_ACTIVE_CITATION":
      return { ...state, activeCitation: action.citation };
    case "ADD_THINKING_STEP": {
      const id = state._stepCounter + 1;
      return {
        ...state,
        _stepCounter: id,
        thinkingSteps: [...state.thinkingSteps, { id, message: action.message }],
      };
    }
    case "SET_TRACE_INFO":
      if (!state.currentAssistantMessageId) {
        return state;
      }
      return {
        ...state,
        traceInfoByMessage: {
          ...state.traceInfoByMessage,
          [state.currentAssistantMessageId]: { traceId: action.traceId, spanId: action.spanId }
        }
      };
    case "SET_INTENT_OVERRIDE":
      return { ...state, intentOverride: action.intentOverride };
    case "SET_CHAT_MODE":
      return { ...state, chatMode: action.mode };
    default:
      return state;
  }
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

const AppStateContext = createContext<AppState>(initialState);
const AppDispatchContext = createContext<Dispatch<AppAction>>(() => { });

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  return (
    <AppStateContext.Provider value={state}>
      <AppDispatchContext.Provider value={dispatch}>
        {children}
      </AppDispatchContext.Provider>
    </AppStateContext.Provider>
  );
}

export function useAppState() {
  return useContext(AppStateContext);
}

export function useAppDispatch() {
  return useContext(AppDispatchContext);
}
