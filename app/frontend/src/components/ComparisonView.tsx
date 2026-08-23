// ============================================================
// ComparisonView.tsx — Side-by-side adaptive vs uniform panes
// Both panes are guaranteed to display the SAME underlying
// frame via the shared frameId key.
// ============================================================
import GridRenderer from './GridRenderer'
import type { FramePayload } from '../lib/types'

interface Props {
  frame:        FramePayload | null
  frameUniform: FramePayload | null
  memorySavedPct: number
  frameId:      number
}

function Badge({ children, color = 'var(--cyan)' }:
  { children: React.ReactNode; color?: string }) {
  return (
    <div style={{
      display: 'inline-flex', alignItems: 'center', gap: 5,
      padding: '3px 10px', borderRadius: 999,
      border: `1px solid ${color}40`,
      background: `${color}12`,
      fontSize: 11, fontFamily: 'var(--text-mono)', color,
    }}>
      {children}
    </div>
  )
}

export default function ComparisonView({ frame, frameUniform, memorySavedPct, frameId }: Props) {
  const adaptiveCells = frame?.cell_count ?? 0
  const uniformCells  = frameUniform?.cell_count ?? 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: 0 }}>

      {/* Comparison header strip */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '8px 16px',
        borderBottom: '1px solid var(--border-dim)',
        background: 'var(--bg-surface)',
      }}>
        <div style={{ display: 'flex', gap: 8 }}>
          <Badge color="var(--red)">Uniform 5cm</Badge>
          <Badge color="var(--green)">Adaptive 5/15/50cm</Badge>
        </div>

        {/* Memory savings — large and readable as spec requires */}
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
          <span style={{
            fontSize: 28, fontFamily: 'var(--text-mono)', fontWeight: 700,
            color: 'var(--green)',
            textShadow: '0 0 16px rgba(0,229,160,0.5)',
          }}>
            {memorySavedPct.toFixed(1)}%
          </span>
          <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>memory saved</span>
        </div>

        <div style={{ display: 'flex', gap: 16, fontSize: 11, fontFamily: 'var(--text-mono)',
          color: 'var(--text-dim)' }}>
          <span>{uniformCells.toLocaleString()} cells</span>
          <span style={{ color: 'var(--green)' }}>→ {adaptiveCells.toLocaleString()} cells</span>
        </div>
      </div>

      {/* Side-by-side panes */}
      <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1,
        background: 'var(--border-dim)' }}>

        {/* LEFT: Uniform baseline */}
        <div style={{ background: 'var(--bg-base)', position: 'relative', overflow: 'hidden' }}>
          <GridRenderer
            key={`u-${frameId}`}
            frame={frameUniform}
            label="Uniform Baseline · 5cm everywhere"
            dim={true}
          />
          {/* Red overlay badge */}
          <div style={{
            position: 'absolute', bottom: 12, left: 12,
            padding: '3px 8px', borderRadius: 4,
            background: 'rgba(255,71,87,0.15)', border: '1px solid rgba(255,71,87,0.3)',
            fontSize: 10, fontFamily: 'var(--text-mono)', color: 'var(--red)',
          }}>
            HIGH MEMORY COST
          </div>
        </div>

        {/* RIGHT: Adaptive */}
        <div style={{ background: 'var(--bg-base)', position: 'relative', overflow: 'hidden' }}>
          <GridRenderer
            key={`a-${frameId}`}
            frame={frame}
            label="Adaptive Ring Grid · Risk-Aware"
          />
          {/* Green savings badge */}
          <div style={{
            position: 'absolute', bottom: 12, right: 12,
            padding: '3px 8px', borderRadius: 4,
            background: 'rgba(0,229,160,0.12)', border: '1px solid rgba(0,229,160,0.3)',
            fontSize: 10, fontFamily: 'var(--text-mono)', color: 'var(--green)',
          }}>
            ↓ {memorySavedPct.toFixed(1)}% CELLS
          </div>
        </div>
      </div>
    </div>
  )
}
