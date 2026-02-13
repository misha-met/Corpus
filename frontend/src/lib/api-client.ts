/**
 * API client for source management endpoints.
 *
 * Provides typed wrappers around fetch calls to the FastAPI backend.
 */

export interface SourceInfo {
  source_id: string;
  summary: string | null;
  source_path: string | null;
  snapshot_path: string | null;
}

export interface SourceListResponse {
  sources: SourceInfo[];
}

export interface IngestResponse {
  source_id: string;
  parents_count: number;
  children_count: number;
  summarized: boolean;
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
  display_page?: string | null;
  header_path?: string;
  chunk_text?: string;
}

/** Response from GET /api/sources/{source_id}/chunk/{chunk_id}. */
export interface ChunkDetailResponse {
  source_id: string;
  chunk_id: string;
  chunk_text: string;
  parent_text?: string | null;
  page_number?: number | null;
  display_page?: string | null;
  header_path: string;
  format: "pdf" | "markdown" | "text";
  source_path?: string | null;
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
    const env =
      typeof process !== "undefined" &&
      typeof process.env?.NEXT_PUBLIC_BACKEND_URL === "string"
        ? process.env.NEXT_PUBLIC_BACKEND_URL
        : "";
    const base = env || "http://127.0.0.1:8000";
    return `${base.replace(/\/$/, "")}/api${path}`;
  }

  async listSources(): Promise<SourceInfo[]> {
    const res = await fetch(`${this.baseUrl}/sources`);
    if (!res.ok) {
      const err: ApiError = await res.json();
      throw new Error(err.error.message);
    }
    const data: SourceListResponse = await res.json();
    return data.sources;
  }

  async getContent(sourceId: string): Promise<SourceContentResponse> {
    const res = await fetch(
      `${this.baseUrl}/sources/${encodeURIComponent(sourceId)}/content`
    );
    if (!res.ok) {
      const err: ApiError = await res.json();
      throw new Error(err.error.message);
    }
    return res.json();
  }

  async deleteSource(sourceId: string): Promise<SourceDeleteResponse> {
    const res = await fetch(
      `${this.baseUrl}/sources/${encodeURIComponent(sourceId)}`,
      { method: "DELETE" }
    );
    if (!res.ok) {
      const err: ApiError = await res.json();
      throw new Error(err.error.message);
    }
    return res.json();
  }

  async ingest(
    filePath: string,
    sourceId: string,
    summarize: boolean = true
  ): Promise<IngestResponse> {
    const res = await fetch(`${this.baseUrl}/sources/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        file_path: filePath,
        source_id: sourceId,
        summarize,
      }),
    });
    if (!res.ok) {
      const err: ApiError = await res.json();
      throw new Error(err.error.message);
    }
    return res.json();
  }

  async uploadDocument(
    file: File,
    sourceId: string,
    summarize: boolean = true
  ): Promise<IngestResponse> {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("source_id", sourceId);
    formData.append("summarize", String(summarize));

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
      const err: ApiError = await res.json();
      throw new Error(err.error.message);
    }
    return res.json();
  }
}

export const sourceApi = new SourceApiClient();
