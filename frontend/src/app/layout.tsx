import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Providers } from "@/components/providers";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "IMSKOS — Intelligent Multi-Source Knowledge Orchestration System",
  description:
    "Production-quality RAG system with adaptive query routing, vector store retrieval, and multi-source knowledge fusion.",
  keywords: [
    "RAG",
    "AI",
    "Knowledge Base",
    "Vector Search",
    "LangChain",
    "Groq",
    "Astra DB",
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
