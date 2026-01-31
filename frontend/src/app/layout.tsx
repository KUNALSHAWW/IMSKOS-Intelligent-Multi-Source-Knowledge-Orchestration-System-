import type { Metadata, Viewport } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Providers } from "@/components/providers";

const inter = Inter({ 
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#fafafa" },
    { media: "(prefers-color-scheme: dark)", color: "#0f0f14" },
  ],
};

export const metadata: Metadata = {
  title: "IMSKOS — Intelligent Multi-Source Knowledge Orchestration System",
  description:
    "Enterprise-grade RAG platform with adaptive query routing, vector store retrieval, and multi-source knowledge fusion. Built for scale and performance.",
  keywords: [
    "RAG",
    "AI",
    "Knowledge Base",
    "Vector Search",
    "LangChain",
    "LangGraph",
    "Groq",
    "Astra DB",
    "Enterprise AI",
    "Semantic Search",
    "Multi-Source Retrieval",
  ],
  authors: [{ name: "IMSKOS Team" }],
  creator: "IMSKOS",
  publisher: "IMSKOS",
  robots: "index, follow",
  openGraph: {
    type: "website",
    locale: "en_US",
    title: "IMSKOS — Intelligent Knowledge Discovery",
    description: "Enterprise-grade RAG platform with adaptive query routing and multi-source fusion.",
    siteName: "IMSKOS",
  },
  twitter: {
    card: "summary_large_image",
    title: "IMSKOS — Intelligent Knowledge Discovery",
    description: "Enterprise-grade RAG platform with adaptive query routing and multi-source fusion.",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning className={`${inter.variable} ${jetbrainsMono.variable}`}>
      <body className={`${inter.className} antialiased`}>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
