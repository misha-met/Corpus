"use client"

import { useEffect, useRef, useState } from "react"
import { cn } from "@/lib/utils"

export interface RainBackgroundProps {
  className?: string
  /** Base number of raindrops */
  count?: number
  /** Rain intensity multiplier */
  intensity?: number
  /** Rain angle in degrees (0 = straight down) */
  angle?: number
  /** Rain color */
  color?: string
  /** Enable lightning flashes */
  lightning?: boolean
}

interface Drop {
  x: number
  y: number
  length: number
  speed: number
  opacity: number
  layer: number
}

const noiseTexture = `url("data:image/svg+xml;utf8,${encodeURIComponent(`
  <svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 120 120">
    <filter id="n">
      <feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves="2" stitchTiles="stitch"/>
    </filter>
    <rect width="100%" height="100%" fill="white" filter="url(#n)"/>
  </svg>
`)}")`

export function RainBackground({
  className,
  count = 150,
  intensity = 1,
  angle = 15,
  color = "rgba(174, 194, 224, 0.5)",
  lightning = true,
}: RainBackgroundProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const flashRef = useRef<HTMLDivElement>(null)
  const [reducedMotion, setReducedMotion] = useState(false)

  useEffect(() => {
    if (typeof window === "undefined") return
    const media = window.matchMedia("(prefers-reduced-motion: reduce)")
    const update = () => setReducedMotion(media.matches)
    update()
    media.addEventListener("change", update)
    return () => media.removeEventListener("change", update)
  }, [])

  useEffect(() => {
    if (reducedMotion) {
      return
    }

    const canvas = canvasRef.current
    const container = containerRef.current
    const flash = flashRef.current
    if (!canvas || !container) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const rect = container.getBoundingClientRect()
    let width = rect.width
    let height = rect.height
    canvas.width = width
    canvas.height = height

    let animationId: number
    const totalDrops = Math.floor(count * intensity)
    const angleRad = (angle * Math.PI) / 180

    const createDrop = (layer: number): Drop => {
      const layerConfig = [
        { speed: 12, length: 15, opacity: 0.2 },
        { speed: 18, length: 20, opacity: 0.35 },
        { speed: 25, length: 28, opacity: 0.5 },
      ][layer]

      return {
        x: Math.random() * (width + 100) - 50,
        y: Math.random() * height - height,
        length: layerConfig.length + Math.random() * 10,
        speed: layerConfig.speed + Math.random() * 5,
        opacity: layerConfig.opacity + Math.random() * 0.1,
        layer,
      }
    }

    const drops: Drop[] = []
    for (let i = 0; i < totalDrops; i++) {
      const layer = i < totalDrops * 0.3 ? 0 : i < totalDrops * 0.6 ? 1 : 2
      drops.push(createDrop(layer))
    }

    const getNextLightningTime = () => Date.now() + 10000 + Math.random() * 15000
    let nextLightning = getNextLightningTime()

    const triggerLightning = () => {
      if (!flash) return
      flash.style.opacity = "0.35"
      setTimeout(() => {
        if (flash) flash.style.opacity = "0.1"
      }, 40)
      setTimeout(() => {
        if (flash) flash.style.opacity = "0"
      }, 110)
      nextLightning = getNextLightningTime()
    }

    const handleResize = () => {
      const r = container.getBoundingClientRect()
      width = r.width
      height = r.height
      canvas.width = width
      canvas.height = height
    }

    const ro = new ResizeObserver(handleResize)
    ro.observe(container)

    const animate = () => {
      ctx.clearRect(0, 0, width, height)

      if (lightning && Date.now() > nextLightning) {
        triggerLightning()
      }

      ctx.strokeStyle = color
      ctx.lineCap = "round"

      for (const drop of drops) {
        drop.y += drop.speed
        drop.x += Math.sin(angleRad) * drop.speed

        if (drop.y > height + 50) {
          drop.y = -drop.length - Math.random() * 100
          drop.x = Math.random() * (width + 100) - 50
        }

        ctx.globalAlpha = drop.opacity
        ctx.lineWidth = drop.layer === 2 ? 1.5 : drop.layer === 1 ? 1 : 0.5
        ctx.beginPath()
        ctx.moveTo(drop.x, drop.y)
        ctx.lineTo(
          drop.x + Math.sin(angleRad) * drop.length,
          drop.y + Math.cos(angleRad) * drop.length,
        )
        ctx.stroke()
      }

      ctx.globalAlpha = 1
      animationId = requestAnimationFrame(animate)
    }

    animationId = requestAnimationFrame(animate)

    return () => {
      cancelAnimationFrame(animationId)
      ro.disconnect()
    }
  }, [count, intensity, angle, color, lightning, reducedMotion])

  return (
    <div
      ref={containerRef}
      className={cn("pointer-events-none absolute inset-0 overflow-hidden", className)}
      style={{
        background: `
          radial-gradient(circle at 50% 20%, rgba(30, 45, 75, 0.12) 0%, transparent 45%),
          linear-gradient(to bottom, #0b0f17 0%, #121827 38%, #171d2b 68%, #11161f 100%)
        `,
      }}
    >
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />

      {lightning && (
        <div
          ref={flashRef}
          className="pointer-events-none absolute inset-0 bg-blue-100 opacity-0 transition-opacity duration-75"
        />
      )}

      <div
        className="pointer-events-none absolute inset-0"
        style={{
          backgroundImage: noiseTexture,
          opacity: 0.035,
          mixBlendMode: "soft-light",
        }}
      />

      <div
        className="pointer-events-none absolute inset-x-0 bottom-0 h-1/3"
        style={{
          background: "linear-gradient(to top, rgba(20, 25, 35, 0.8) 0%, transparent 100%)",
        }}
      />

      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse at center, transparent 0%, transparent 40%, rgba(8,10,15,0.7) 100%)",
        }}
      />
    </div>
  )
}

export default RainBackground