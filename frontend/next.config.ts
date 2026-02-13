import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /**
   * Proxy API requests to the FastAPI backend during development.
   * This avoids CORS issues and mirrors the production setup where
   * both frontend and backend share the same origin.
   */
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://127.0.0.1:8000/api/:path*",
      },
    ];
  },

  /**
   * Disable response buffering for proxied streaming responses.
   * Next.js rewrites pass through headers from the upstream, but
   * we also need to ensure the dev server doesn't buffer chunks.
   */
  async headers() {
    return [
      {
        source: "/api/:path*",
        headers: [
          { key: "X-Accel-Buffering", value: "no" },
          { key: "Cache-Control", value: "no-cache" },
        ],
      },
    ];
  },
};

export default nextConfig;
