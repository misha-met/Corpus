"use client";

import {
  createContext,
  useContext,
  useReducer,
  type Dispatch,
  type ReactNode,
} from "react";

// ---------------------------------------------------------------------------
// State shape
// ---------------------------------------------------------------------------

export interface AppState {
  /** Whether a chat request is currently in-flight */
  isBusy: boolean;
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
}

const initialState: AppState = {
  isBusy: false,
  statusMessage: "",
  errorMessage: "",
  isLockBusy: false,
  lastIntent: null,
  lastSources: [],
};

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

export type AppAction =
  | { type: "CHAT_START" }
  | { type: "CHAT_FINISH" }
  | { type: "SET_STATUS"; status: string }
  | { type: "SET_ERROR"; message: string; isLockBusy?: boolean }
  | { type: "CLEAR_ERROR" }
  | { type: "SET_INTENT"; intent: string; confidence: number; method: string }
  | { type: "SET_SOURCES"; sourceIds: string[] };

// ---------------------------------------------------------------------------
// Reducer
// ---------------------------------------------------------------------------

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case "CHAT_START":
      return {
        ...state,
        isBusy: true,
        statusMessage: "Sending...",
        errorMessage: "",
        isLockBusy: false,
        lastIntent: null,
        lastSources: [],
      };
    case "CHAT_FINISH":
      return {
        ...state,
        isBusy: false,
        statusMessage: "",
      };
    case "SET_STATUS":
      return { ...state, statusMessage: action.status };
    case "SET_ERROR":
      return {
        ...state,
        isBusy: false,
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
