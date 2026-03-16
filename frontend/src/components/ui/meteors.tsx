"use client"

import { useMemo } from "react"
import { cn } from "@/lib/utils"

export interface MeteorsProps {
  className?: string
  /** Number of meteors */
  count?: number
  /** Meteor angle in degrees (215 = diagonal down-left) */
  angle?: number
  /** Meteor color */
  color?: string
  /** Tail gradient color */
  tailColor?: string
}

interface MeteorData {
  id: number
  left: number
  delay: number
  duration: number
}

function seededUnit(seed: number): number {
  const x = Math.sin(seed * 12.9898) * 43758.5453
  return x - Math.floor(x)
}

export function Meteors({
  className,
  count = 12,
  angle = 215,
  color = "#ffffff",
  tailColor = "#ffffff",
}: MeteorsProps) {
  const meteors = useMemo<MeteorData[]>(
    () =>
      Array.from({ length: count }, (_, i) => ({
        id: i,
        left: i * (100 / count),
        delay: seededUnit(i + 1) * 10,
        duration: 6 + seededUnit((i + 1) * 7.13) * 10,
      })),
    [count],
  )

  return (
    <div className={cn("pointer-events-none absolute inset-0", className)}>
      <style>{`
        @keyframes meteor-fall {
          0% {
            transform: rotate(${angle}deg) translateX(0);
            opacity: 1;
          }
          70% {
            opacity: 1;
          }
          100% {
            transform: rotate(${angle}deg) translateX(-100vmax);
            opacity: 0;
          }
        }
      `}</style>

      {/* Night-sky base — extends 2px beyond bounds to prevent edge banding */}
      <div className="pointer-events-none absolute" style={{ inset: "-2px", background: "#0a0a0a" }} />

      {/* Meteors */}
      {meteors.map(meteor => (
        <span
          key={meteor.id}
          className="absolute rounded-full"
          style={{
            top: "-40px",
            left: `${meteor.left}%`,
            width: "2px",
            height: "2px",
            backgroundColor: color,
            boxShadow: `0 0 4px 1px rgba(255,255,255,0.6)`,
            animation: `meteor-fall ${meteor.duration}s linear infinite`,
            animationDelay: `${meteor.delay}s`,
          }}
        >
          {/* Tail */}
          <span
            className="absolute top-1/2 -translate-y-1/2"
            style={{
              left: "100%",
              width: "80px",
              height: "1px",
              background: `linear-gradient(to right, ${tailColor}, transparent)`,
            }}
          />
        </span>
      ))}

    </div>
  )
}

export default Meteors
