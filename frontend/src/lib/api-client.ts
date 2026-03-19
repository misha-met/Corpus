/**
 * API client for source management endpoints.
 *
 * Provides typed wrappers around fetch calls to the FastAPI backend.
 */

import { getBackendApiBase as _getBackendApiBase } from "./backend-url";

export interface SourceInfo {
  source_id: string;
  summary: string | null;
  source_path: string | null;
  snapshot_path: string | null;
  citation_reference?: string | null;
  page_offset?: number;
  source_size_bytes?: number | null;
  content_size_bytes?: number | null;
}

export interface SourceListResponse {
  sources: SourceInfo[];
}

export interface IngestResponse {
  source_id: string;
  parents_count: number;
  children_count: number;
  summarized: boolean;
  geotag_ner?: NERDiagnostics | null;
  peopletag_ner?: NERDiagnostics | null;
}

export interface NERDiagnostics {
  ner_available: boolean;
  method: string;
  warning?: string | null;
}

export interface SourceContentResponse {
  source_id: string;
  content: string;
  content_source: "original" | "snapshot" | "summary";
  format: "pdf" | "markdown" | "text";
}

export interface SourceDeleteResponse {
  source_id: string;
  deleted: boolean;
}

/** A single citation entry emitted via the CITATIONS: stream line. */
export interface CitationEntry {
  index: number;
  source_id: string;
  chunk_id: string;
  page_number?: number | null;
  display_page?: string | null;
  header_path?: string;
  chunk_text: string;
}

/** Payload passed to CitationViewerModal on citation click. */
export interface CitationPayload {
  source_id: string;
  chunk_id?: string;
  page_number?: number | null;
  start_page?: number | null;
  end_page?: number | null;
  display_page?: string | null;
  header_path?: string;
  chunk_text?: string;
}

export interface ChunkDetailResponse {
  source_id: string;
  chunk_id: string;
  chunk_text: string;
  parent_text?: string | null;
  page_number?: number | null;
  start_page?: number | null;
  end_page?: number | null;
  display_page?: string | null;
  header_path: string;
  format: "pdf" | "markdown" | "text";
  source_path?: string | null;
}

export interface ChunkBatchItem {
  source_id: string;
  chunk_id: string;
  chunk_text: string;
  page_number?: number | null;
  start_page?: number | null;
  end_page?: number | null;
  display_page?: string | null;
  header_path: string;
  format: "pdf" | "markdown" | "text";
  source_path?: string | null;
}

export interface ChunkBatchResponse {
  chunks: ChunkBatchItem[];
}

export interface GeoMentionDetail {
  id: string;
  source_id: string;
  chunk_id: string;
  matched_input: string;
  confidence: number;
  method: string;
}

export interface GeoMentionGroup {
  place_name: string;
  geonameid: number;
  lat: number;
  lon: number;
  mention_count: number;
  max_confidence: number;
  matched_inputs: string[];
  source_ids: string[];
  chunk_ids: string[];
  mention_ids: string[];
  mentions: GeoMentionDetail[];
}

export interface GeoMentionsResponse {
  count: number;
  mentions: GeoMentionGroup[];
}

export interface PersonMention {
  id: string;
  source_id: string;
  chunk_id: string;
  raw_name: string;
  canonical_name: string;
  confidence: number;
  method: string;
  role_hint?: string | null;
  context_snippet: string;
}

export interface PersonSummary {
  canonical_name: string;
  mention_count: number;
  source_count: number;
  source_ids: string[];
  variants: string[];
  roles: string[];
  avg_confidence: number;
}

export interface PeopleListResponse {
  count: number;
  people: PersonSummary[];
}

export interface PersonMentionsResponse {
  canonical_name: string;
  count: number;
  mentions: PersonMention[];
}

export interface PeopleMergeResponse {
  source_canonical_name: string;
  target_canonical_name: string;
  merged_count: number;
}

export interface ApiError {
  error: {
    code: string;
    message: string;
  };
}

class SourceApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = "/api") {
    this.baseUrl = baseUrl;
  }

  /**
   * Parse an error response that may or may not be JSON.
   * Returns a human-readable error message.
   */
  private async parseErrorResponse(res: Response): Promise<string> {
    // 502/503/504 means Next.js couldn't reach the backend at all
    if (res.status === 502 || res.status === 503 || res.status === 504) {
      return "Backend not reachable — make sure the server is running on port 8000.";
    }
    if (res.status === 500) {
      try {
        const body = await res.json();
        return body?.error?.message ?? "Internal server error — something went wrong on the backend.";
      } catch {
        return "Internal server error — something went wrong on the backend.";
      }
    }
    try {
      const body = await res.json();
      return body?.error?.message ?? `HTTP ${res.status}`;
    } catch {
      // Response body is not valid JSON (e.g. plain "Internal Server Error")
      try {
        const text = await res.text();
        return text || `HTTP ${res.status}`;
      } catch {
        return `HTTP ${res.status}`;
      }
    }
  }

  /**
   * Resolve the backend URL for long-running operations that must
   * bypass the Next.js dev proxy to avoid its ~30s timeout.
   */
  private getDirectBackendUrl(path: string): string {
    return `${_getBackendApiBase()}${path}`;
  }

  async listSources(): Promise<SourceInfo[]> {
    const res = await fetch(`${this.baseUrl}/sources`);
    if (!res.ok) {
      const message = await this.parseErrorResponse(res);
      throw new Error(message);
    }
    const data: SourceListResponse = await res.json();
    return data.sources;
  }

  async getContent(sourceId: string): Promise<SourceContentResponse> {
    const res = await fetch(
      `${this.baseUrl}/sources/${encodeURIComponent(sourceId)}/content`
    );
    if (!res.ok) {
      const message = await this.parseErrorResponse(res);
      throw new Error(message);
    }
    return res.json();
  }

  async deleteSource(sourceId: string): Promise<SourceDeleteResponse> {
    const res = await fetch(
      `${this.baseUrl}/sources/${encodeURIComponent(sourceId)}`,
      { method: "DELETE" }
    );
    if (!res.ok) {
      const message = await this.parseErrorResponse(res);
      throw new Error(message);
    }
    return res.json();
  }

  async ingest(
    filePath: string,
    sourceId: string,
    summarize: boolean = true,
    geotag: boolean = false,
    peopletag: boolean = false,
    citationReference?: string,
  ): Promise<IngestResponse> {
    const res = await fetch(`${this.baseUrl}/sources/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        file_path: filePath,
        source_id: sourceId,
        summarize,
        geotag,
        peopletag,
        citation_reference: citationReference,
      }),
    });
    if (!res.ok) {
      const message = await this.parseErrorResponse(res);
      throw new Error(message);
    }
    return res.json();
  }

  async uploadDocument(
    file: File,
    sourceId: string,
    summarize: boolean = true,
    pageOffset?: number,
    geotag: boolean = false,
    peopletag: boolean = false,
    citationReference?: string,
  ): Promise<IngestResponse> {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("source_id", sourceId);
    formData.append("summarize", String(summarize));
    formData.append("geotag", String(geotag));
    formData.append("peopletag", String(peopletag));
    formData.append("page_offset", String(pageOffset ?? 1));
    if (citationReference) {
      formData.append("citation_reference", citationReference);
    }

    // Call the backend directly to bypass the Next.js dev proxy
    // which has a ~30s timeout — upload+ingest can take minutes.
    const url = this.getDirectBackendUrl("/sources/upload");
    const res = await fetch(url, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const message = await this.parseErrorResponse(res);
      throw new Error(message);
    }
    return res.json();
  }

  async getChunk(
    sourceId: string,
    chunkId: string
  ): Promise<ChunkDetailResponse> {
    const res = await fetch(
      `${this.baseUrl}/sources/${encodeURIComponent(sourceId)}/chunk/${encodeURIComponent(chunkId)}`
    );
    if (!res.ok) {
      const message = await this.parseErrorResponse(res);
      throw new Error(message);
    }
    return res.json();
  }

  async getChunks(
    sourceId: string,
    chunkIds: string[]
  ): Promise<ChunkBatchResponse> {
    if (!chunkIds.length) return { chunks: [] };
    const res = await fetch(
      `${this.baseUrl}/sources/${encodeURIComponent(sourceId)}/chunks?ids=${encodeURIComponent(chunkIds.join(","))}`
    );
    if (!res.ok) {
      const message = await this.parseErrorResponse(res);
      throw new Error(message);
    }
    return res.json();
  }

  async getGeoMentions(
    sourceId?: string,
    minConfidence: number = 0.75,
    limit: number = 1000,
    offset: number = 0,
    detailed: boolean = true,
    sourceIds?: string[],
    q?: string,
  ): Promise<GeoMentionsResponse> {
    const params = new URLSearchParams({
      min_confidence: String(minConfidence),
      limit: String(limit),
      offset: String(offset),
      detailed: String(detailed),
    });
    if (sourceId) {
      params.set("source_id", sourceId);
    }
    if (sourceIds !== undefined) {
      if (sourceIds.length === 0) {
        // Preserve explicit-empty contract for backend filtering.
        params.append("source_ids", "");
      } else {
        const seen = new Set<string>();
        for (const raw of sourceIds) {
          const sid = String(raw).trim();
          if (!sid || seen.has(sid)) continue;
          seen.add(sid);
          params.append("source_ids", sid);
        }
      }
    }
    if (typeof q === "string" && q.trim().length > 0) {
      params.set("q", q.trim());
    }

    const res = await fetch(`${this.baseUrl}/geo/mentions?${params.toString()}`);
    if (!res.ok) {
      const message = await this.parseErrorResponse(res);
      throw new Error(message);
    }
    return res.json();
  }

  async deleteGeoMention(mentionId: string): Promise<void> {
    const res = await fetch(
      `${this.baseUrl}/geo/mentions/${encodeURIComponent(mentionId)}`,
      { method: "DELETE" }
    );
    if (!res.ok) {
      const message = await this.parseErrorResponse(res);
      throw new Error(message);
    }
  }

  async getPeople(
    sourceId?: string,
    minConfidence: number = 0.0,
    q?: string,
    limit: number = 200,
    offset: number = 0,
    sourceIds?: string[],
  ): Promise<PeopleListResponse> {
    const params = new URLSearchParams({
      min_confidence: String(minConfidence),
      limit: String(limit),
      offset: String(offset),
    });
    if (sourceId) {
      params.set("source_id", sourceId);
    }
    if (q && q.trim().length > 0) {
      params.set("q", q.trim());
    }
    if (sourceIds !== undefined) {
      if (sourceIds.length === 0) {
        params.append("source_ids", "");
      } else {
        const seen = new Set<string>();
        for (const raw of sourceIds) {
          const sid = String(raw).trim();
          if (!sid || seen.has(sid)) continue;
          seen.add(sid);
          params.append("source_ids", sid);
        }
      }
    }

    const res = await fetch(`${this.baseUrl}/people?${params.toString()}`);
    if (!res.ok) {
      const message = await this.parseErrorResponse(res);
      throw new Error(message);
    }
    return res.json();
  }

  async getPeopleMentions(
    canonicalName: string,
    sourceId?: string,
    minConfidence: number = 0.0,
    limit: number = 1000,
    offset: number = 0,
    sourceIds?: string[],
  ): Promise<PersonMentionsResponse> {
    const params = new URLSearchParams({
      canonical_name: canonicalName,
      min_confidence: String(minConfidence),
      limit: String(limit),
      offset: String(offset),
    });
    if (sourceId) {
      params.set("source_id", sourceId);
    }
    if (sourceIds !== undefined) {
      if (sourceIds.length === 0) {
        params.append("source_ids", "");
      } else {
        const seen = new Set<string>();
        for (const raw of sourceIds) {
          const sid = String(raw).trim();
          if (!sid || seen.has(sid)) continue;
          seen.add(sid);
          params.append("source_ids", sid);
        }
      }
    }

    const res = await fetch(`${this.baseUrl}/people/mentions?${params.toString()}`);
    if (!res.ok) {
      const message = await this.parseErrorResponse(res);
      throw new Error(message);
    }
    return res.json();
  }

  async deletePeopleMention(mentionId: string): Promise<void> {
    const res = await fetch(
      `${this.baseUrl}/people/mentions/${encodeURIComponent(mentionId)}`,
      { method: "DELETE" },
    );
    if (!res.ok) {
      const message = await this.parseErrorResponse(res);
      throw new Error(message);
    }
  }

  async mergePeopleCanonical(
    sourceCanonicalName: string,
    targetCanonicalName: string,
  ): Promise<PeopleMergeResponse> {
    const source = sourceCanonicalName.trim();
    const target = targetCanonicalName.trim();
    if (!source || !target) {
      throw new Error("Source and target canonical names are required.");
    }

    const res = await fetch(`${this.baseUrl}/people/merge`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_canonical_name: source,
        target_canonical_name: target,
      }),
    });
    if (!res.ok) {
      const message = await this.parseErrorResponse(res);
      throw new Error(message);
    }
    return res.json();
  }
}

export const sourceApi = new SourceApiClient();

// ---------------------------------------------------------------------------
// QueryStreaming — legacy SSE-based RAG query streaming
// ---------------------------------------------------------------------------

export interface StreamEvent {
  event: string;
  data: Record<string, unknown>;
}

export interface QueryStreamOptions {
  sourceIds?: string[];
  citationsEnabled?: boolean;
  signal?: AbortSignal;
}

/**
 * Stream a RAG query as plain SSE events from /api/query.
 *
 * Yields objects with `event` (string) and `data` (parsed JSON object) for
 * each SSE block received. Recognised event names include: "status",
 * "intent", "sources", "citations", "token", "error", "complete".
 */
export async function* queryStreaming(
  query: string,
  options: QueryStreamOptions = {},
): AsyncGenerator<StreamEvent, void, unknown> {
  const { sourceIds, citationsEnabled = true, signal } = options;

  let res: Response;
  try {
    res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        stream: true,
        source_ids: sourceIds ?? [],
        citations_enabled: citationsEnabled,
      }),
      signal,
    });
  } catch (err) {
    if ((err as Error)?.name === "AbortError") return;
    yield { event: "error", data: { error: "Backend not reachable — make sure the server is running on port 8000." } };
    return;
  }

  if (!res.ok || !res.body) {
    let msg: string;
    if (res.status === 502 || res.status === 503 || res.status === 504) {
      msg = "Backend not reachable — make sure the server is running on port 8000.";
    } else if (res.status === 500) {
      msg = "Internal server error — something went wrong on the backend.";
    } else {
      msg = `HTTP ${res.status}`;
    }
    yield { event: "error", data: { error: msg } };
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  function* parseBuffer(buf: string): Generator<StreamEvent> {
    const blocks = buf.split(/\r?\n\r?\n/);
    for (const block of blocks) {
      if (!block.trim() || block.trimStart().startsWith(":")) continue;
      const lines = block.split(/\r?\n/);
      let evt = "message";
      let dat = "";
      for (const line of lines) {
        if (line.startsWith("event:")) evt = line.slice(6).trim();
        else if (line.startsWith("data:")) dat = line.slice(5).trimStart();
      }
      if (!dat) continue;
      let parsed: Record<string, unknown>;
      try { parsed = JSON.parse(dat) as Record<string, unknown>; } catch { parsed = {}; }
      yield { event: evt, data: parsed };
    }
  }

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split(/\r?\n\r?\n/);
      buffer = blocks.pop() ?? "";
      const rejoined = blocks.join("\n\n");
      yield* parseBuffer(rejoined);
    }
    // Flush remaining buffer
    if (buffer.trim()) {
      yield* parseBuffer(buffer);
    }
  } finally {
    reader.releaseLock();
  }
}
