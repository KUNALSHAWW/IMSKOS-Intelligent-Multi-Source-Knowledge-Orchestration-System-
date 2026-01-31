import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import Home from "@/app/page";

// Mock the API client
vi.mock("@/lib/api", () => ({
  apiClient: {
    query: vi.fn(),
    health: vi.fn(),
  },
}));

import { apiClient } from "@/lib/api";

const mockQueryResponse = {
  id: "mock-query-0001",
  response: "This is a deterministic mock answer from IMSKOS scaffold.",
  sources: [
    {
      source_id: "doc-1",
      similarity_score: 0.92,
      snippet: "Example matched text...",
      url: "/storage/docs/doc1.pdf",
    },
  ],
  metrics: { elapsed_ms: 75, tokens: 34, embedding_ms: 15, retrieval_ms: 35, llm_ms: 25 },
  routing_reason: "mock: scaffold route to vector (demo)",
};

function renderWithProviders(component: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>{component}</QueryClientProvider>
  );
}

describe("Home Page", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders the IMSKOS title", () => {
    renderWithProviders(<Home />);
    // New UI has "IMSKOS" in both header and footer
    const imskosElements = screen.getAllByText("IMSKOS");
    expect(imskosElements.length).toBeGreaterThan(0);
  });

  it("renders the query input textarea", () => {
    renderWithProviders(<Home />);
    expect(screen.getByLabelText("Query input")).toBeInTheDocument();
  });

  it("renders example query buttons", () => {
    renderWithProviders(<Home />);
    expect(screen.getByText("What are the types of agent memory?")).toBeInTheDocument();
  });

  it("disables execute button when query is empty", () => {
    renderWithProviders(<Home />);
    const button = screen.getByLabelText("Execute query");
    expect(button).toBeDisabled();
  });

  it("enables execute button when query has content", () => {
    renderWithProviders(<Home />);
    const textarea = screen.getByLabelText("Query input");
    fireEvent.change(textarea, { target: { value: "Test query" } });
    
    const button = screen.getByLabelText("Execute query");
    expect(button).not.toBeDisabled();
  });

  it("fills textarea when clicking example query", () => {
    renderWithProviders(<Home />);
    const exampleButton = screen.getByText("What are the types of agent memory?");
    fireEvent.click(exampleButton);
    
    const textarea = screen.getByLabelText("Query input") as HTMLTextAreaElement;
    expect(textarea.value).toBe("What are the types of agent memory?");
  });

  it("submits query and displays results", async () => {
    vi.mocked(apiClient.query).mockResolvedValue(mockQueryResponse);
    
    renderWithProviders(<Home />);
    
    const textarea = screen.getByLabelText("Query input");
    fireEvent.change(textarea, { target: { value: "Test query" } });
    
    const button = screen.getByLabelText("Execute query");
    fireEvent.click(button);
    
    await waitFor(() => {
      expect(screen.getByText(mockQueryResponse.response)).toBeInTheDocument();
    });
    
    // Check that sources are displayed - new UI says "Retrieved Sources"
    expect(screen.getByText("Retrieved Sources")).toBeInTheDocument();
    expect(screen.getByText(/92% match/)).toBeInTheDocument();
    
    // Check metrics - new UI shows number without "ms" suffix (it's in the label)
    expect(screen.getByText("75")).toBeInTheDocument();
  });

  it("displays error when query fails", async () => {
    vi.mocked(apiClient.query).mockRejectedValue(new Error("Network error"));
    
    renderWithProviders(<Home />);
    
    const textarea = screen.getByLabelText("Query input");
    fireEvent.change(textarea, { target: { value: "Test query" } });
    
    const button = screen.getByLabelText("Execute query");
    fireEvent.click(button);

    // New UI shows "Query Failed" instead of "Error:"
    await waitFor(() => {
      expect(screen.getByText("Query Failed")).toBeInTheDocument();
    });
  });

  it("renders feature cards", () => {
    renderWithProviders(<Home />);
    expect(screen.getByText("Adaptive Routing")).toBeInTheDocument();
    expect(screen.getByText("Vector Storage")).toBeInTheDocument();
    expect(screen.getByText("Multi-Source Fusion")).toBeInTheDocument();
  });
});
