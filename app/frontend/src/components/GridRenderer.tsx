// ============================================================
// GridRenderer.tsx — Canvas-based variable-resolution 2.5D renderer
// Cell size drives rectangle size directly from backend data.
// ============================================================
import { useEffect, useRef } from 'react'
import { CLASS_COLORS } from '../lib/constants'
import type { FramePayload } from '../lib/types'

interface Props {
  frame:  FramePayload | null
  label?: string
  dim?:   boolean
}

const WORLD_RANGE = 100  // metres half-extent
const BG = '#070d14'

function hexToRgba(hex: string, alpha: number) {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return `rgba(${r},${g},${b},${alpha})`
}

export default function GridRenderer({ frame, label, dim = false }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const W = canvas.width
    const H = canvas.height

    // Clear
    ctx.fillStyle = BG
    ctx.fillRect(0, 0, W, H)

    if (!frame || frame.cells.length === 0) {
      // Loading shimmer
      const grad = ctx.createLinearGradient(0, 0, W, 0)
      grad.addColorStop(0, '#0c1521')
      grad.addColorStop(0.5, '#111d2e')
      grad.addColorStop(1, '#0c1521')
      ctx.fillStyle = grad
      ctx.fillRect(0, 0, W, H)
      ctx.fillStyle = 'rgba(0,200,255,0.06)'
      ctx.font = '12px JetBrains Mono, monospace'
      ctx.textAlign = 'center'
      ctx.fillText('AWAITING STREAM…', W / 2, H / 2)
      return
    }

    // World-to-canvas transform
    // Centre is origin (sensor), Y-axis flipped
    const scale = Math.min(W, H) / (WORLD_RANGE * 2)
    const cx = W / 2
    const cy = H / 2

    const toCanvasX = (wx: number) => cx + wx * scale
    const toCanvasY = (wy: number) => cy - wy * scale   // flip Y

    // Draw origin rings (range rings)
    ctx.save()
    ctx.strokeStyle = 'rgba(0,200,255,0.06)'
    ctx.lineWidth = 1
    for (const r of [10, 30, 60, 100]) {
      ctx.beginPath()
      ctx.arc(cx, cy, r * scale, 0, Math.PI * 2)
      ctx.stroke()
    }
    ctx.restore()

    // Draw cells
    for (const cell of frame.cells) {
      const px = toCanvasX(cell.x - cell.size / 2)
      const py = toCanvasY(cell.y + cell.size / 2)
      const pw = cell.size * scale
      const ph = cell.size * scale

      const baseColor = CLASS_COLORS[cell.class_id] ?? CLASS_COLORS[255]

      // Height shading: raise brightness for elevated terrain
      const heightAlpha = 0.55 + Math.min(0.45, Math.abs(cell.height) * 0.4)
      ctx.fillStyle = hexToRgba(baseColor, heightAlpha)
      ctx.fillRect(px, py, Math.max(pw, 1), Math.max(ph, 1))
    }

    // Sensor origin cross
    ctx.save()
    ctx.strokeStyle = 'rgba(0,200,255,0.7)'
    ctx.lineWidth = 1.5
    ctx.beginPath()
    ctx.moveTo(cx - 6, cy); ctx.lineTo(cx + 6, cy)
    ctx.moveTo(cx, cy - 6); ctx.lineTo(cx, cy + 6)
    ctx.stroke()
    ctx.restore()

    // Dim overlay for non-active pane
    if (dim) {
      ctx.fillStyle = 'rgba(2,4,8,0.5)'
      ctx.fillRect(0, 0, W, H)
    }
  }, [frame, dim])

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <canvas
        ref={canvasRef}
        width={480}
        height={480}
        style={{
          width: '100%',
          height: '100%',
          display: 'block',
          borderRadius: 'var(--radius-md)',
        }}
      />
      {label && (
        <div style={{
          position: 'absolute', top: 10, left: 12,
          fontSize: 11, fontFamily: 'var(--text-mono)',
          color: 'var(--text-secondary)',
          letterSpacing: '0.08em', textTransform: 'uppercase',
        }}>
          {label}
        </div>
      )}
      {frame && (
        <div style={{
          position: 'absolute', bottom: 10, right: 12,
          fontSize: 10, fontFamily: 'var(--text-mono)',
          color: 'var(--text-dim)',
        }}>
          {frame.cell_count.toLocaleString()} cells
        </div>
      )}
    </div>
  )
}
