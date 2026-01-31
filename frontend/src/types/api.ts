// API Response Types

export type QuerySource = "auto" | "vector" | "wikipedia" | "web";

export interface QueryOptions {
  top_k?: number;
  temperature?: number;
  hyde?: boolean;
}

export interface QueryRequest {
  query: string;
  user_id?: string;
  source?: QuerySource;
  options?: QueryOptions;
}

export interface SourceResult {
  source_id: string;
  similarity_score: number;
  snippet: string;
  url?: string;
}

export interface QueryMetrics {
  elapsed_ms: number;
  tokens: number;
  embedding_ms?: number;
  retrieval_ms?: number;
  llm_ms?: number;
}

export interface QueryResponse {
  id: string;
  response: string;
  sources: SourceResult[];
  metrics: QueryMetrics;
  routing_reason: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  environment: string;
  mock_mode: Record<string, boolean>;
  timestamp: string;
}

export interface ApiError {
  detail: string;
  status_code: number;
}
