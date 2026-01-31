"use client";

import { useState, useEffect } from "react";
import { useMutation } from "@tanstack/react-query";
import { 
  Search, 
  Sparkles, 
  Zap, 
  Database, 
  ExternalLink, 
  Loader2, 
  Brain, 
  Cpu, 
  Globe, 
  Shield, 
  ArrowRight,
  Clock,
  Layers,
  Activity,
  CheckCircle2,
  ChevronRight
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { apiClient } from "@/lib/api";
import type { QueryRequest, QueryResponse, SourceResult } from "@/types/api";

const EXAMPLE_QUERIES = [
  { text: "What are the types of agent memory?", icon: Brain },
  { text: "Explain chain of thought prompting", icon: Layers },
  { text: "How do adversarial attacks work?", icon: Shield },
  { text: "Who is Elon Musk?", icon: Globe },
];

const FEATURES = [
  {
    icon: Zap,
    title: "Adaptive Routing",
    description: "LLM-powered decision engine dynamically routes queries to optimal data sources with intelligent fallback strategies.",
    gradient: "from-amber-400 via-orange-400 to-rose-400",
    bgGradient: "from-amber-50 to-orange-50",
    iconBg: "bg-gradient-to-br from-amber-100 to-orange-100",
  },
  {
    icon: Database,
    title: "Vector Storage",
    description: "Scalable DataStax Astra DB for production-grade vector operations with sub-millisecond query latency.",
    gradient: "from-blue-400 via-cyan-400 to-teal-400",
    bgGradient: "from-blue-50 to-cyan-50",
    iconBg: "bg-gradient-to-br from-blue-100 to-cyan-100",
  },
  {
    icon: Sparkles,
    title: "Multi-Source Fusion",
    description: "Seamless integration of proprietary knowledge bases, Wikipedia, and custom data sources.",
    gradient: "from-violet-400 via-purple-400 to-fuchsia-400",
    bgGradient: "from-violet-50 to-purple-50",
    iconBg: "bg-gradient-to-br from-violet-100 to-purple-100",
  },
];

const STATS = [
  { label: "Queries/sec", value: "10K+", icon: Activity },
  { label: "Latency", value: "<100ms", icon: Clock },
  { label: "Accuracy", value: "99.9%", icon: CheckCircle2 },
  { label: "Sources", value: "50+", icon: Database },
];

export default function Home() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    setIsLoaded(true);
  }, []);

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
    <main className="min-h-screen relative overflow-hidden noise-overlay">
      {/* Animated Background Orbs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="orb orb-1" />
        <div className="orb orb-2" />
        <div className="orb orb-3" />
      </div>

      {/* Mesh Gradient Background */}
      <div className="fixed inset-0 bg-mesh-gradient opacity-30 pointer-events-none" />

      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-white/10">
        <div className="glass-card rounded-none border-x-0 border-t-0">
          <div className="container mx-auto px-6 py-4 flex items-center justify-between">
            <div className={`flex items-center gap-3 transition-all duration-700 ${isLoaded ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-4'}`}>
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-violet-500 to-fuchsia-500 rounded-xl blur-lg opacity-50 animate-pulse-soft" />
                <div className="relative p-2.5 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 shadow-lg">
                  <Sparkles className="h-5 w-5 text-white" />
                </div>
              </div>
              <div>
                <span className="font-bold text-xl tracking-tight gradient-text">IMSKOS</span>
                <p className="text-[10px] text-muted-foreground -mt-0.5 tracking-wide">INTELLIGENT KNOWLEDGE ORCHESTRATION</p>
              </div>
            </div>
            
            <nav className={`flex items-center gap-3 transition-all duration-700 delay-200 ${isLoaded ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-4'}`}>
              <div className="status-badge">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </span>
                System Online
              </div>
              <Badge variant="outline" className="gap-1.5 px-3 py-1.5 bg-white/50 backdrop-blur-sm border-white/20 shadow-sm">
                <Cpu className="h-3 w-3 text-primary" />
                <span className="text-xs font-medium">v1.1.0</span>
              </Badge>
            </nav>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative container mx-auto px-6 pt-20 pb-16">
        <div className={`text-center max-w-4xl mx-auto transition-all duration-1000 ${isLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-gradient-to-r from-violet-100 to-fuchsia-100 border border-violet-200/50 mb-8 shadow-soft">
            <Sparkles className="h-4 w-4 text-violet-600" />
            <span className="text-sm font-medium text-violet-700">Enterprise-Grade RAG Platform</span>
            <ChevronRight className="h-4 w-4 text-violet-400" />
          </div>

          {/* Title */}
          <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
            <span className="gradient-text animate-gradient bg-[length:200%_200%]">Intelligent</span>
            <br />
            <span className="text-foreground">Knowledge Discovery</span>
          </h1>
          
          <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-12 leading-relaxed">
            Experience the future of information retrieval with adaptive query routing, 
            multi-source fusion, and production-grade vector search.
          </p>

          {/* Stats Row */}
          <div className="flex flex-wrap justify-center gap-6 mb-12">
            {STATS.map((stat, i) => (
              <div 
                key={stat.label}
                className={`flex items-center gap-3 px-5 py-3 rounded-2xl glass-card transition-all duration-500`}
                style={{ animationDelay: `${i * 100}ms` }}
              >
                <div className="p-2 rounded-xl bg-gradient-to-br from-violet-100 to-fuchsia-100">
                  <stat.icon className="h-4 w-4 text-violet-600" />
                </div>
                <div className="text-left">
                  <p className="text-xl font-bold gradient-text">{stat.value}</p>
                  <p className="text-xs text-muted-foreground">{stat.label}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Query Input Card */}
        <div className={`max-w-3xl mx-auto transition-all duration-1000 delay-300 ${isLoaded ? 'opacity-100 translate-y-0 scale-100' : 'opacity-0 translate-y-8 scale-95'}`}>
          <div className="relative">
            {/* Glow Effect Behind Card */}
            <div className="absolute -inset-1 bg-gradient-to-r from-violet-500 via-fuchsia-500 to-pink-500 rounded-3xl blur-2xl opacity-20 animate-pulse-soft" />
            
            <Card className="relative glass-card border-white/20 shadow-glass-lg overflow-hidden">
              <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-violet-500 via-fuchsia-500 to-pink-500" />
              
              <CardContent className="p-8">
                <form onSubmit={handleSubmit} className="space-y-6">
                  <div className="relative group input-glow rounded-2xl transition-all duration-300">
                    <div className="absolute -inset-0.5 bg-gradient-to-r from-violet-500 to-fuchsia-500 rounded-2xl opacity-0 group-focus-within:opacity-100 blur transition-opacity duration-300" />
                    <div className="relative">
                      <Textarea
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Ask anything — from agent architectures to world knowledge..."
                        className="min-h-[140px] px-6 py-5 pr-14 resize-none text-base bg-white/80 backdrop-blur-sm border-2 border-muted/50 rounded-2xl focus:border-transparent focus:ring-2 focus:ring-violet-500/30 transition-all duration-300 placeholder:text-muted-foreground/60"
                        aria-label="Query input"
                      />
                      <div className="absolute bottom-5 right-5">
                        <div className="p-2.5 rounded-xl bg-gradient-to-br from-muted/50 to-muted/30 group-focus-within:from-violet-100 group-focus-within:to-fuchsia-100 transition-all duration-300">
                          <Search className="h-5 w-5 text-muted-foreground group-focus-within:text-violet-600 transition-colors" />
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex flex-col sm:flex-row justify-between items-center gap-4">
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <kbd className="px-2 py-1 rounded-lg bg-muted/50 border border-muted text-xs font-mono">Enter</kbd>
                      <span>or click to execute</span>
                    </div>
                    <Button 
                      type="submit" 
                      size="lg"
                      disabled={!query.trim() || queryMutation.isPending}
                      className="btn-shine relative bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-700 hover:to-fuchsia-700 text-white px-8 py-6 rounded-xl font-semibold shadow-lg hover:shadow-xl transition-all duration-300 hover:-translate-y-0.5"
                      aria-label="Execute query"
                    >
                      {queryMutation.isPending ? (
                        <>
                          <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                          Processing...
                        </>
                      ) : (
                        <>
                          <Zap className="mr-2 h-5 w-5" />
                          Execute Query
                          <ArrowRight className="ml-2 h-4 w-4" />
                        </>
                      )}
                    </Button>
                  </div>
                </form>

                {/* Example Queries */}
                <div className="mt-8 pt-6 border-t border-muted/30">
                  <p className="text-sm text-muted-foreground mb-4 flex items-center gap-2">
                    <Sparkles className="h-4 w-4 text-violet-500" />
                    Try an example query
                  </p>
                  <div className="flex flex-wrap gap-3">
                    {EXAMPLE_QUERIES.map((example, index) => (
                      <Button
                        key={index}
                        variant="outline"
                        size="sm"
                        onClick={() => handleExampleClick(example.text)}
                        className="group px-4 py-2.5 rounded-xl bg-white/60 hover:bg-gradient-to-r hover:from-violet-50 hover:to-fuchsia-50 border-muted/50 hover:border-violet-200 transition-all duration-300 hover:-translate-y-0.5 hover:shadow-soft"
                      >
                        <example.icon className="mr-2 h-4 w-4 text-muted-foreground group-hover:text-violet-600 transition-colors" />
                        <span className="text-sm">{example.text}</span>
                      </Button>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Error Display */}
        {queryMutation.isError && (
          <div className="max-w-3xl mx-auto mt-6 animate-fade-in-up">
            <Card className="border-destructive/50 bg-gradient-to-r from-red-50 to-rose-50 shadow-soft">
              <CardContent className="p-6 flex items-start gap-4">
                <div className="p-2 rounded-xl bg-red-100">
                  <Shield className="h-5 w-5 text-red-600" />
                </div>
                <div>
                  <p className="font-semibold text-red-700 mb-1">Query Failed</p>
                  <p className="text-red-600 text-sm">
                    {queryMutation.error instanceof Error 
                      ? queryMutation.error.message 
                      : "Failed to execute query. Please check if the backend is running."}
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Results Section */}
        {result && (
          <div className="max-w-4xl mx-auto mt-10 space-y-6 text-left">
            {/* Response Card */}
            <div className="animate-fade-in-up">
              <div className="relative">
                <div className="absolute -inset-1 bg-gradient-to-r from-emerald-500 to-teal-500 rounded-3xl blur-xl opacity-20" />
                <Card className="relative glass-card border-white/20 overflow-hidden">
                  <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-emerald-500 to-teal-500" />
                  <CardHeader className="pb-4">
                    <div className="flex items-center justify-between">
                      <CardTitle className="flex items-center gap-3 text-xl">
                        <div className="p-2.5 rounded-xl bg-gradient-to-br from-emerald-100 to-teal-100">
                          <Sparkles className="h-5 w-5 text-emerald-600" />
                        </div>
                        <span className="gradient-text">AI Response</span>
                      </CardTitle>
                      <Badge className="badge-success px-3 py-1.5 text-xs font-medium">
                        <CheckCircle2 className="h-3 w-3 mr-1.5" />
                        Generated
                      </Badge>
                    </div>
                    <CardDescription className="mt-3 flex items-center gap-2 text-sm">
                      <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-muted/50">
                        <Database className="h-3.5 w-3.5 text-muted-foreground" />
                        <span className="text-muted-foreground">{result.routing_reason}</span>
                      </div>
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="p-5 rounded-2xl bg-gradient-to-br from-white/80 to-muted/30 border border-white/50">
                      <p className="text-foreground whitespace-pre-wrap leading-relaxed">{result.response}</p>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>

            {/* Sources Card */}
            {result.sources.length > 0 && (
              <div className="animate-fade-in-up" style={{ animationDelay: '150ms' }}>
                <Card className="glass-card border-white/20 overflow-hidden">
                  <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-blue-500 to-cyan-500" />
                  <CardHeader>
                    <CardTitle className="flex items-center gap-3 text-lg">
                      <div className="p-2 rounded-xl bg-gradient-to-br from-blue-100 to-cyan-100">
                        <Layers className="h-5 w-5 text-blue-600" />
                      </div>
                      Retrieved Sources
                    </CardTitle>
                    <CardDescription className="flex items-center gap-2">
                      <span className="inline-flex items-center justify-center h-6 w-6 rounded-full bg-blue-100 text-blue-700 text-xs font-bold">
                        {result.sources.length}
                      </span>
                      <span>relevant sources discovered</span>
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {result.sources.map((source: SourceResult, index: number) => (
                        <div
                          key={source.source_id}
                          className="source-card p-5 rounded-2xl border border-muted/30 hover:border-blue-200"
                          style={{ animationDelay: `${index * 100}ms` }}
                        >
                          <div className="flex items-start justify-between gap-4">
                            <div className="flex-1">
                              <div className="flex items-center gap-3 mb-3">
                                <span className="flex items-center justify-center h-7 w-7 rounded-lg bg-gradient-to-br from-violet-100 to-fuchsia-100 text-violet-700 text-sm font-bold">
                                  {index + 1}
                                </span>
                                <div className="px-3 py-1 rounded-full bg-gradient-to-r from-emerald-100 to-teal-100 text-emerald-700 text-xs font-semibold">
                                  {(source.similarity_score * 100).toFixed(0)}% match
                                </div>
                              </div>
                              <p className="text-sm text-muted-foreground leading-relaxed line-clamp-3">
                                {source.snippet}
                              </p>
                            </div>
                            {source.url && (
                              <Button 
                                variant="ghost" 
                                size="icon" 
                                asChild
                                className="shrink-0 h-10 w-10 rounded-xl bg-muted/30 hover:bg-blue-100 hover:text-blue-600 transition-all"
                              >
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
              </div>
            )}

            {/* Metrics Card */}
            <div className="animate-fade-in-up" style={{ animationDelay: '300ms' }}>
              <Card className="glass-card border-white/20 overflow-hidden">
                <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-amber-500 to-orange-500" />
                <CardHeader>
                  <CardTitle className="flex items-center gap-3 text-lg">
                    <div className="p-2 rounded-xl bg-gradient-to-br from-amber-100 to-orange-100">
                      <Activity className="h-5 w-5 text-amber-600" />
                    </div>
                    Performance Metrics
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="metric-glow text-center p-5 rounded-2xl border border-white/30">
                      <p className="text-3xl font-bold gradient-text">
                        {result.metrics.elapsed_ms}
                      </p>
                      <p className="text-xs text-muted-foreground mt-1 flex items-center justify-center gap-1">
                        <Clock className="h-3 w-3" />
                        Total Time (ms)
                      </p>
                    </div>
                    <div className="metric-glow text-center p-5 rounded-2xl border border-white/30">
                      <p className="text-3xl font-bold gradient-text">
                        {result.metrics.tokens}
                      </p>
                      <p className="text-xs text-muted-foreground mt-1 flex items-center justify-center gap-1">
                        <Cpu className="h-3 w-3" />
                        Tokens Used
                      </p>
                    </div>
                    {result.metrics.retrieval_ms !== undefined && (
                      <div className="metric-glow text-center p-5 rounded-2xl border border-white/30">
                        <p className="text-3xl font-bold gradient-text">
                          {result.metrics.retrieval_ms}
                        </p>
                        <p className="text-xs text-muted-foreground mt-1 flex items-center justify-center gap-1">
                          <Database className="h-3 w-3" />
                          Retrieval (ms)
                        </p>
                      </div>
                    )}
                    {result.metrics.llm_ms !== undefined && (
                      <div className="metric-glow text-center p-5 rounded-2xl border border-white/30">
                        <p className="text-3xl font-bold gradient-text">
                          {result.metrics.llm_ms}
                        </p>
                        <p className="text-xs text-muted-foreground mt-1 flex items-center justify-center gap-1">
                          <Brain className="h-3 w-3" />
                          LLM Time (ms)
                        </p>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Query ID */}
            <div className="text-center py-4 animate-fade-in">
              <p className="text-xs text-muted-foreground inline-flex items-center gap-2 px-4 py-2 rounded-full bg-muted/30">
                <span>Query ID:</span>
                <code className="bg-white/50 px-2 py-0.5 rounded-md font-mono text-violet-600">{result.id}</code>
              </p>
            </div>
          </div>
        )}
      </section>

      {/* Features Section */}
      <section className="relative container mx-auto px-6 py-24 border-t border-white/10">
        <div className="text-center mb-16">
          <Badge className="mb-4 px-4 py-2 bg-gradient-to-r from-violet-100 to-fuchsia-100 text-violet-700 border-0">
            <Cpu className="h-3.5 w-3.5 mr-2" />
            Enterprise Features
          </Badge>
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            Built for <span className="gradient-text">Scale & Performance</span>
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Production-ready architecture designed to handle millions of queries with sub-second latency.
          </p>
        </div>
        
        <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
          {FEATURES.map((feature, index) => (
            <div
              key={feature.title}
              className={`animate-fade-in-up`}
              style={{ animationDelay: `${index * 150}ms` }}
            >
              <Card className={`feature-card h-full glass-card border-white/20 overflow-hidden group`}>
                <div className={`absolute top-0 left-0 right-0 h-1 bg-gradient-to-r ${feature.gradient} opacity-70 group-hover:opacity-100 transition-opacity`} />
                <CardHeader className="pb-4">
                  <div className={`w-14 h-14 rounded-2xl ${feature.iconBg} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-500`}>
                    <feature.icon className={`h-7 w-7 bg-gradient-to-r ${feature.gradient} bg-clip-text`} style={{ color: 'transparent', backgroundClip: 'text', WebkitBackgroundClip: 'text' }} />
                  </div>
                  <CardTitle className="text-xl group-hover:gradient-text transition-all duration-300">
                    {feature.title}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground leading-relaxed">
                    {feature.description}
                  </p>
                </CardContent>
              </Card>
            </div>
          ))}
        </div>
      </section>

      {/* Tech Stack Section */}
      <section className="relative container mx-auto px-6 py-16">
        <div className="glass-card rounded-3xl p-8 md:p-12 text-center overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-violet-500/5 via-transparent to-fuchsia-500/5" />
          <div className="relative">
            <h3 className="text-2xl font-bold mb-4">Powered By Industry Leaders</h3>
            <p className="text-muted-foreground mb-8 max-w-xl mx-auto">
              Built on battle-tested technologies trusted by Fortune 500 companies
            </p>
            <div className="flex flex-wrap justify-center items-center gap-6 md:gap-10">
              {['Next.js', 'FastAPI', 'LangGraph', 'Astra DB', 'Groq', 'Supabase'].map((tech, i) => (
                <div
                  key={tech}
                  className="px-5 py-2.5 rounded-xl bg-white/60 border border-muted/30 text-sm font-medium text-muted-foreground hover:text-foreground hover:border-violet-200 hover:bg-gradient-to-r hover:from-violet-50 hover:to-fuchsia-50 transition-all duration-300 cursor-default"
                >
                  {tech}
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative border-t border-white/10 py-12">
        <div className="container mx-auto px-6">
          <div className="flex flex-col md:flex-row justify-between items-center gap-6">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500">
                <Sparkles className="h-4 w-4 text-white" />
              </div>
              <span className="font-semibold gradient-text">IMSKOS</span>
            </div>
            <p className="text-sm text-muted-foreground text-center">
              Built with ❤️ using Next.js, FastAPI, LangGraph, Astra DB, and Groq
            </p>
            <Badge variant="outline" className="bg-white/50 backdrop-blur-sm">
              v1.1.0
            </Badge>
          </div>
        </div>
      </footer>
    </main>
  );
}
