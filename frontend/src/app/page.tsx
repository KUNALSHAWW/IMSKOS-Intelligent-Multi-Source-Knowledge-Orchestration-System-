"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Search, Sparkles, Zap, Database, ExternalLink, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { apiClient } from "@/lib/api";
import type { QueryRequest, QueryResponse, SourceResult } from "@/types/api";

const EXAMPLE_QUERIES = [
  "What are the types of agent memory?",
  "Explain chain of thought prompting techniques",
  "How do adversarial attacks work on large language models?",
  "Who is Elon Musk?",
];

export default function Home() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<QueryResponse | null>(null);

  const queryMutation = useMutation({
    mutationFn: (request: QueryRequest) => apiClient.query(request),
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    
    queryMutation.mutate({
      query: query.trim(),
      source: "auto",
      options: {
        top_k: 5,
        temperature: 0,
        hyde: false,
      },
    });
  };

  const handleExampleClick = (example: string) => {
    setQuery(example);
  };

  return (
    <main className="min-h-screen bg-gradient-to-b from-background to-muted/20">
      {/* Header */}
      <header className="border-b bg-background/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sparkles className="h-6 w-6 text-primary" />
            <span className="font-bold text-xl">IMSKOS</span>
            <Badge variant="secondary" className="ml-2">
              Demo
            </Badge>
          </div>
          <nav className="flex items-center gap-4">
            <Badge variant="outline" className="gap-1">
              <Zap className="h-3 w-3" />
              Mock Mode
            </Badge>
          </nav>
        </div>
      </header>

      {/* Hero Section */}
      <section className="container mx-auto px-4 py-16 text-center">
        <h1 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
          IMSKOS — Demo
        </h1>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto mb-8">
          Intelligent Multi-Source Knowledge Orchestration System. Ask anything and watch 
          the adaptive query router find the best sources.
        </p>

        {/* Query Input */}
        <Card className="max-w-3xl mx-auto">
          <CardContent className="pt-6">
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="relative">
                <Textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Ask a question about AI, technology, or anything else..."
                  className="min-h-[120px] pr-12 resize-none text-base"
                  aria-label="Query input"
                />
                <div className="absolute bottom-3 right-3">
                  <Search className="h-5 w-5 text-muted-foreground" />
                </div>
              </div>
              
              <div className="flex justify-between items-center gap-4">
                <div className="text-sm text-muted-foreground">
                  Press Enter or click Execute
                </div>
                <Button 
                  type="submit" 
                  size="lg"
                  disabled={!query.trim() || queryMutation.isPending}
                  aria-label="Execute query"
                >
                  {queryMutation.isPending ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Zap className="mr-2 h-4 w-4" />
                      Execute Query
                    </>
                  )}
                </Button>
              </div>
            </form>

            {/* Example Queries */}
            <div className="mt-6 pt-4 border-t">
              <p className="text-sm text-muted-foreground mb-3">Try an example:</p>
              <div className="flex flex-wrap gap-2">
                {EXAMPLE_QUERIES.map((example, index) => (
                  <Button
                    key={index}
                    variant="outline"
                    size="sm"
                    onClick={() => handleExampleClick(example)}
                    className="text-xs"
                  >
                    {example}
                  </Button>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Error Display */}
        {queryMutation.isError && (
          <Card className="max-w-3xl mx-auto mt-6 border-destructive">
            <CardContent className="pt-6">
              <p className="text-destructive">
                Error: {queryMutation.error instanceof Error 
                  ? queryMutation.error.message 
                  : "Failed to execute query. Please check if the backend is running."}
              </p>
            </CardContent>
          </Card>
        )}

        {/* Results Section */}
        {result && (
          <div className="max-w-4xl mx-auto mt-8 space-y-6 text-left">
            {/* Response Card */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Sparkles className="h-5 w-5 text-primary" />
                  Response
                </CardTitle>
                <CardDescription className="flex items-center gap-2">
                  <Badge variant="outline" className="gap-1">
                    <Database className="h-3 w-3" />
                    {result.routing_reason}
                  </Badge>
                </CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-foreground whitespace-pre-wrap">{result.response}</p>
              </CardContent>
            </Card>

            {/* Sources Card */}
            {result.sources.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Sources</CardTitle>
                  <CardDescription>
                    {result.sources.length} relevant source(s) found
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {result.sources.map((source: SourceResult, index: number) => (
                      <div
                        key={source.source_id}
                        className="p-4 rounded-lg border bg-muted/50"
                      >
                        <div className="flex items-start justify-between gap-4">
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-2">
                              <Badge variant="secondary">
                                #{index + 1}
                              </Badge>
                              <Badge variant="success">
                                {(source.similarity_score * 100).toFixed(0)}% match
                              </Badge>
                            </div>
                            <p className="text-sm text-muted-foreground line-clamp-3">
                              {source.snippet}
                            </p>
                          </div>
                          {source.url && (
                            <Button variant="ghost" size="icon" asChild>
                              <a
                                href={source.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                aria-label={`Open source ${source.source_id}`}
                              >
                                <ExternalLink className="h-4 w-4" />
                              </a>
                            </Button>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Metrics Card */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Performance Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center p-3 rounded-lg bg-muted/50">
                    <p className="text-2xl font-bold text-primary">
                      {result.metrics.elapsed_ms}ms
                    </p>
                    <p className="text-xs text-muted-foreground">Total Time</p>
                  </div>
                  <div className="text-center p-3 rounded-lg bg-muted/50">
                    <p className="text-2xl font-bold text-primary">
                      {result.metrics.tokens}
                    </p>
                    <p className="text-xs text-muted-foreground">Tokens</p>
                  </div>
                  {result.metrics.retrieval_ms && (
                    <div className="text-center p-3 rounded-lg bg-muted/50">
                      <p className="text-2xl font-bold text-primary">
                        {result.metrics.retrieval_ms}ms
                      </p>
                      <p className="text-xs text-muted-foreground">Retrieval</p>
                    </div>
                  )}
                  {result.metrics.llm_ms && (
                    <div className="text-center p-3 rounded-lg bg-muted/50">
                      <p className="text-2xl font-bold text-primary">
                        {result.metrics.llm_ms}ms
                      </p>
                      <p className="text-xs text-muted-foreground">LLM</p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Query ID */}
            <div className="text-center">
              <p className="text-xs text-muted-foreground">
                Query ID: <code className="bg-muted px-1 rounded">{result.id}</code>
              </p>
            </div>
          </div>
        )}
      </section>

      {/* Features Section */}
      <section className="container mx-auto px-4 py-16 border-t">
        <h2 className="text-2xl font-bold text-center mb-8">Features</h2>
        <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg">
                <Zap className="h-5 w-5 text-yellow-500" />
                Adaptive Routing
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                LLM-powered decision engine that dynamically routes queries to optimal data sources.
              </p>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg">
                <Database className="h-5 w-5 text-blue-500" />
                Vector Storage
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Scalable DataStax Astra DB for production-grade vector operations.
              </p>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg">
                <Sparkles className="h-5 w-5 text-purple-500" />
                Multi-Source Fusion
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Seamless integration of proprietary knowledge and public sources like Wikipedia.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t py-8">
        <div className="container mx-auto px-4 text-center">
          <p className="text-sm text-muted-foreground">
            Built with ❤️ using Next.js, FastAPI, LangGraph, Astra DB, and Groq
          </p>
          <p className="text-xs text-muted-foreground mt-2">
            IMSKOS v1.1.0 — Scaffold Demo
          </p>
        </div>
      </footer>
    </main>
  );
}
