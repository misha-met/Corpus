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
}

export interface SourceDeleteResponse {
  source_id: string;
  deleted: boolean;
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
}

export const sourceApi = new SourceApiClient();
