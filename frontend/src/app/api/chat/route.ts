/**
 * Phase 2 — Streaming proxy for /api/chat → FastAPI backend.
 *
 * The Next.js dev-server rewrite proxy (next.config.ts) buffers the SSE
 * response body before forwarding it, causing all tokens to appear at once.
 * A Route Handler returns a raw ReadableStream, which Next.js pipes through
 * without buffering, giving true token-by-token streaming.
 *
 * assistant-ui's useChatRuntime points at /api/chat (this file).
 * This handler forwards the body to http://127.0.0.1:8000/api/chat and
 * streams the SSE response back to the browser unchanged.
 */

const FASTAPI_CHAT_URL = "http://127.0.0.1:8000/api/chat";

export async function POST(req: Request): Promise<Response> {
  const body = await req.text(); // forward the body as-is (JSON string)

  const upstream = await fetch(FASTAPI_CHAT_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      // Forward any auth headers the client sent
      ...(req.headers.get("authorization")
        ? { Authorization: req.headers.get("authorization")! }
        : {}),
    },
    body,
    // @ts-expect-error — Node.js fetch needs this to disable its own buffering
    duplex: "half",
  });

  if (!upstream.ok || !upstream.body) {
    const text = await upstream.text();
    return new Response(text, { status: upstream.status });
  }

  // Pass the ReadableStream through directly — no buffering.
  return new Response(upstream.body, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "x-vercel-ai-ui-message-stream": "v1",
      "X-Accel-Buffering": "no",
    },
  });
}
