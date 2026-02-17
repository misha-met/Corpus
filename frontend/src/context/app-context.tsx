/**
 * AppContext — custom application state for DH Notebook.
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
  /** Citation list from the latest query's CitationListEvent */
  citations: Citation[];
  /** The citation the user most recently clicked — read by CitationViewerModal */
  activeCitation: Citation | null;
}

const initialState: AppState = {
  statusMessage: "",
  errorMessage: "",
  isLockBusy: false,
  lastIntent: null,
  lastSources: [],
  citations: [],
  activeCitation: null,
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
  | { type: "SET_ACTIVE_CITATION"; citation: Citation | null };

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
        citations: [],
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
      return { ...state, citations: action.citations };
    case "SET_ACTIVE_CITATION":
      return { ...state, activeCitation: action.citation };
    default:
      return state;
  }
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

const AppStateContext = createContext<AppState>(initialState);
const AppDispatchContext = createContext<Dispatch<AppAction>>(() => {});

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
