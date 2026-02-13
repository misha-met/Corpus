/**
 * Custom fetch wrapper for the AI SDK's useChat hook.
 *
 * Intercepts 429 (LOCK_BUSY) responses and throws a typed error
 * that the UI can catch to show a "server busy" message instead
 * of a generic error.
 */

export class LockBusyError extends Error {
  constructor(message: string = "Another query is already in progress") {
    super(message);
    this.name = "LockBusyError";
  }
}

export class ApiError extends Error {
  code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = "ApiError";
    this.code = code;
  }
}

/**
 * Custom fetch function for useChat that intercepts error responses.
 *
 * Usage:
 * ```tsx
 * const { messages } = useChat({
 *   api: "/api/chat",
 *   fetch: apiFetch,
 * });
 * ```
 */
export const apiFetch: typeof fetch = async (input, init) => {
  const response = await fetch(input, init);

  if (response.status === 429) {
    let message = "Another query is already in progress";
    try {
      const body = await response.json();
      message = body?.error?.message ?? message;
    } catch {
      // Use default message
    }
    throw new LockBusyError(message);
  }

  if (!response.ok && response.status !== 200) {
    let code = "UNKNOWN";
    let message = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      code = body?.error?.code ?? code;
      message = body?.error?.message ?? message;
    } catch {
      // Use default message
    }
    throw new ApiError(code, message);
  }

  return response;
};
