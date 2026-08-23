// ============================================================
// MetricsPanel.tsx — Live FPS / latency / memory readout
// ============================================================
import { TARGET_FPS } from '../lib/constants'
import type { MetricsPayload } from '../lib/types'

interface Props {
  metrics: MetricsPayload | null
}

function FpsColor(fps: number): string {
  if (fps >= TARGET_FPS * 0.8) return 'var(--green)'
  if (fps >= TARGET_FPS * 0.5) return 'var(--amber)'
  return 'var(--red)'
}

function StatRow({ label, value, unit, mono = true }:
  { label: string; value: string | number; unit?: string; mono?: boolean }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
      padding: '5px 0', borderBottom: '1px solid var(--border-dim)' }}>
      <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>{label}</span>
      <span style={{
        fontSize: 13,
        fontFamily: mono ? 'var(--text-mono)' : 'inherit',
        color: 'var(--text-primary)',
      }}>
        {value}{unit && <span style={{ fontSize: 10, color: 'var(--text-dim)', marginLeft: 3 }}>{unit}</span>}
      </span>
    </div>
  )
}

function LatencyBar({ label, value, max, color }:
  { label: string; value: number; max: number; color: string }) {
  const pct = Math.min(100, (value / max) * 100)
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
        <span style={{ fontSize: 10, color: 'var(--text-dim)' }}>{label}</span>
        <span style={{ fontSize: 10, fontFamily: 'var(--text-mono)', color }}>{value.toFixed(1)}ms</span>
      </div>
      <div style={{ height: 3, background: 'var(--border-dim)', borderRadius: 2 }}>
        <div style={{
          height: '100%', width: `${pct}%`, borderRadius: 2,
          background: color, transition: 'width 0.3s ease',
        }} />
      </div>
    </div>
  )
}

export default function MetricsPanel({ metrics }: Props) {
  if (!metrics) {
    return (
      <div style={{ padding: 16, opacity: 0.4 }}>
        <div style={{ fontSize: 11, color: 'var(--text-dim)', textAlign: 'center', paddingTop: 32 }}>
          Awaiting metrics…
        </div>
      </div>
    )
  }

  const fpsColor = FpsColor(metrics.fps)
  const totalMs = metrics.latency_ms.total || 1

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

      {/* FPS Hero */}
      <div style={{ textAlign: 'center', padding: '12px 0' }}>
        <div style={{ fontSize: 10, color: 'var(--text-dim)', letterSpacing: '0.1em',
          textTransform: 'uppercase', marginBottom: 4 }}>
          Achieved FPS
        </div>
        <div style={{
          fontSize: 48, fontFamily: 'var(--text-mono)', fontWeight: 600,
          color: fpsColor, lineHeight: 1,
          textShadow: `0 0 20px ${fpsColor}60`,
          transition: 'color 0.3s ease',
        }}>
          {metrics.fps.toFixed(1)}
        </div>
        <div style={{ fontSize: 10, color: 'var(--text-dim)', marginTop: 4 }}>
          target {TARGET_FPS} fps
        </div>
      </div>

      {/* Pipeline Stage Latencies */}
      <div>
        <div style={{ fontSize: 10, color: 'var(--text-dim)', letterSpacing: '0.1em',
          textTransform: 'uppercase', marginBottom: 10 }}>
          Pipeline Latency
        </div>
        <LatencyBar label="Classify (Perception)" value={metrics.latency_ms.classify}
          max={totalMs} color="var(--cyan)" />
        <LatencyBar label="Project (Grid Engine)" value={metrics.latency_ms.project}
          max={totalMs} color="var(--green)" />
        <LatencyBar label="Serialize (JSON)" value={metrics.latency_ms.serialize}
          max={totalMs} color="var(--amber)" />
        <div style={{ paddingTop: 4, display: 'flex', justifyContent: 'space-between',
          borderTop: '1px solid var(--border-dim)' }}>
          <span style={{ fontSize: 10, color: 'var(--text-secondary)' }}>Total (end-to-end)</span>
          <span style={{ fontSize: 12, fontFamily: 'var(--text-mono)', color: 'var(--text-primary)' }}>
            {metrics.latency_ms.total.toFixed(1)}<span style={{ fontSize: 10, color: 'var(--text-dim)', marginLeft: 2 }}>ms</span>
          </span>
        </div>
      </div>

      {/* Stats */}
      <div>
        <div style={{ fontSize: 10, color: 'var(--text-dim)', letterSpacing: '0.1em',
          textTransform: 'uppercase', marginBottom: 6 }}>
          Frame Stats
        </div>
        <StatRow label="Points / Frame" value={metrics.point_count.toLocaleString()} />
        <StatRow label="Memory Saved" value={`${metrics.memory_saved_pct.toFixed(1)}%`} />
        <StatRow label="Window Samples" value={metrics.window_samples} />
        <StatRow label="Backend" value={metrics.backend} mono={false} />
      </div>

    </div>
  )
}
