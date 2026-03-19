"use client";

import { X } from "lucide-react";

interface OverlayPanelProps {
  open: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
}

/**
 * Shared animated overlay used by both the Map and People panels.
 * Renders a scrim, an animated card container with rounded borders,
 * a top-gradient cap, a close button, and the supplied children.
 */
export function OverlayPanel({
  open,
  onClose,
  title,
  children,
}: OverlayPanelProps) {
  return (
    <div
      role="dialog"
      aria-label={title}
      className="absolute inset-0 z-20 p-2 sm:p-3 md:p-4"
      style={{
        opacity: open ? 1 : 0,
        transform: open ? "translateY(0) scale(1)" : "translateY(10px) scale(0.992)",
        filter: open ? "blur(0)" : "blur(1.75px)",
        pointerEvents: open ? "auto" : "none",
        transition:
          "opacity 420ms cubic-bezier(0.22,1,0.36,1), transform 520ms cubic-bezier(0.22,1,0.36,1), filter 420ms ease",
        willChange: open ? "opacity, transform, filter" : "auto",
        transformOrigin: "center center",
      }}
    >
      {/* Scrim */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "linear-gradient(180deg, rgba(7,10,14,0.82) 0%, rgba(7,10,14,0.90) 100%), " +
            "radial-gradient(120% 80% at 50% 50%, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0) 68%)",
          opacity: open ? 1 : 0,
          transition: "opacity 520ms cubic-bezier(0.22,1,0.36,1)",
        }}
      />

      {/* Card wrapper (secondary animation layer) */}
      <div
        className="relative h-full w-full"
        style={{
          opacity: open ? 1 : 0,
          transform: open ? "translateY(0) scale(1)" : "translateY(18px) scale(0.972)",
          transition:
            "opacity 460ms cubic-bezier(0.22,1,0.36,1), transform 560ms cubic-bezier(0.22,1,0.36,1)",
          willChange: open ? "opacity, transform" : "auto",
        }}
      >
        {/* Card */}
        <div className="relative h-full w-full overflow-hidden rounded-2xl border border-white/20 bg-black/92 shadow-[0_20px_60px_rgba(0,0,0,0.66),inset_0_1px_0_rgba(255,255,255,0.08)]">
          <div className="pointer-events-none absolute inset-x-0 top-0 z-20 h-16 bg-gradient-to-b from-black/65 via-black/30 to-transparent" />

          <button
            onClick={onClose}
            className="absolute top-3 right-3 z-30 p-2 rounded-full bg-black/75 text-white/80 hover:text-white hover:bg-black border border-white/20 shadow-md"
            title={`Close ${title}`}
            aria-label={`Close ${title}`}
          >
            <X className="w-4 h-4" />
          </button>

          {children}
        </div>
      </div>
    </div>
  );
}
