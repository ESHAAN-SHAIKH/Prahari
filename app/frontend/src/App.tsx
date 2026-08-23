// ============================================================
// App.tsx — Main application layout (TASK-015→018)
// Premium defence-grade dark UI for PRAHARI-Lite dashboard
// ============================================================
import { useState } from 'react'
import { api } from './lib/apiClient'
import { useLiveData } from './hooks/useLiveData'
import GridRenderer from './components/GridRenderer'
import ComparisonView from './components/ComparisonView'
import MetricsPanel from './components/MetricsPanel'
import Legend from './components/Legend'
import type { ViewMode } from './lib/types'

function LiveDot({ connected }: { connected: boolean }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <div style={{
        width: 7, height: 7, borderRadius: '50%',
        background: connected ? 'var(--green)' : 'var(--red)',
        boxShadow: connected ? '0 0 8px var(--green)' : 'none',
        animation: connected ? 'pulse-dot 2s ease-in-out infinite' : 'none',
      }} />
      <span style={{ fontSize: 10, fontFamily: 'var(--text-mono)',
        color: connected ? 'var(--green)' : 'var(--red)',
        letterSpacing: '0.08em', textTransform: 'uppercase' }}>
        {connected ? 'Live' : 'Disconnected'}
      </span>
    </div>
  )
}

function TabBtn({ active, onClick, children }:
  { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '5px 14px', borderRadius: 'var(--radius-sm)',
        border: active ? '1px solid var(--border-accent)' : '1px solid transparent',
        background: active ? 'var(--cyan-dim)' : 'transparent',
        color: active ? 'var(--cyan)' : 'var(--text-secondary)',
        fontSize: 12, fontFamily: 'var(--text-mono)',
        cursor: 'pointer', transition: 'all 0.15s ease',
        letterSpacing: '0.05em',
      }}>
      {children}
    </button>
  )
}

function ControlBtn({ onClick, label, active, color = 'var(--cyan)' }:
  { onClick: () => void; label: string; active?: boolean; color?: string }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '5px 12px', borderRadius: 'var(--radius-sm)',
        border: `1px solid ${active ? color : 'var(--border-dim)'}`,
        background: active ? `${color}18` : 'transparent',
        color: active ? color : 'var(--text-secondary)',
        fontSize: 11, fontFamily: 'var(--text-mono)',
        cursor: 'pointer', transition: 'all 0.15s ease',
      }}>
      {label}
    </button>
  )
}

export default function App() {
  const [isPlaying, setIsPlaying] = useState(false)
  const [viewMode, setViewMode] = useState<ViewMode>('adaptive')
  const [backend, setBackend] = useState<'groundtruth' | 'model'>('groundtruth')
  const [speed, setSpeed] = useState(1.0)

  const { frame, frameUniform, metrics, connected, frameId } = useLiveData(isPlaying)

  const handleStart = async () => {
    await api.playbackStart()
    setIsPlaying(true)
  }

  const handleStop = async () => {
    await api.playbackStop()
    setIsPlaying(false)
  }

  const handleSpeedChange = async (x: number) => {
    await api.playbackSpeed(x)
    setSpeed(x)
  }

  const handleBackendToggle = async () => {
    const next = backend === 'groundtruth' ? 'model' : 'groundtruth'
    await api.switchBackend(next)
    setBackend(next)
  }

  const savedPct = metrics?.memory_saved_pct ?? 32.4

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>

      {/* ── TOP HEADER ─────────────────────────────────────── */}
      <header style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 20px', height: 52,
        background: 'var(--bg-surface)',
        borderBottom: '1px solid var(--border-dim)',
        flexShrink: 0,
      }}>
        {/* Wordmark */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 28, height: 28, borderRadius: 6,
            background: 'linear-gradient(135deg, var(--cyan) 0%, #0050ff 100%)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 13, fontWeight: 700, color: '#000',
          }}>P</div>
          <div>
            <div style={{ fontSize: 14, fontWeight: 600, letterSpacing: '-0.02em' }}>PRAHARI-Lite</div>
            <div style={{ fontSize: 9, color: 'var(--text-dim)', letterSpacing: '0.1em',
              textTransform: 'uppercase' }}>Adaptive Perception System</div>
          </div>
        </div>

        {/* View tabs */}
        <div style={{ display: 'flex', gap: 4, padding: '3px', borderRadius: 'var(--radius-sm)',
          background: 'var(--bg-elevated)' }}>
          <TabBtn active={viewMode === 'adaptive'}   onClick={() => setViewMode('adaptive')}>Adaptive</TabBtn>
          <TabBtn active={viewMode === 'uniform'}    onClick={() => setViewMode('uniform')}>Uniform</TabBtn>
          <TabBtn active={viewMode === 'comparison'} onClick={() => setViewMode('comparison')}>Compare</TabBtn>
        </div>

        {/* Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {/* Speed pills */}
          <div style={{ display: 'flex', gap: 3 }}>
            {[0.5, 1, 2].map(s => (
              <ControlBtn key={s} onClick={() => handleSpeedChange(s)}
                label={`${s}×`} active={speed === s} />
            ))}
          </div>

          {/* Backend toggle */}
          <ControlBtn
            onClick={handleBackendToggle}
            label={backend === 'groundtruth' ? 'GT Backend' : 'Neural Backend'}
            color={backend === 'model' ? 'var(--amber)' : 'var(--cyan)'}
            active
          />

          {/* Play/Stop */}
          {isPlaying
            ? <ControlBtn onClick={handleStop}  label="⏸ Pause" color="var(--amber)" active />
            : <ControlBtn onClick={handleStart} label="▶ Play"  color="var(--green)"  active />}

          <LiveDot connected={connected} />
        </div>
      </header>

      {/* ── MAIN BODY ──────────────────────────────────────── */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

        {/* ── CENTRE: Grid View ────────────────────────────── */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden',
          background: 'var(--bg-base)' }}>

          {viewMode === 'comparison' ? (
            <ComparisonView
              frame={frame} frameUniform={frameUniform}
              memorySavedPct={savedPct} frameId={frameId}
            />
          ) : (
            <div style={{ flex: 1, padding: 16, position: 'relative' }}>
              <GridRenderer
                key={`${viewMode}-${frameId}`}
                frame={viewMode === 'adaptive' ? frame : frameUniform}
                label={viewMode === 'adaptive'
                  ? 'Adaptive Ring Grid · 5 / 15 / 50 cm tiers'
                  : 'Uniform Baseline · 5 cm everywhere'}
              />

              {/* Frame counter badge */}
              {frame && (
                <div style={{
                  position: 'absolute', top: 24, right: 24,
                  padding: '4px 10px', borderRadius: 'var(--radius-sm)',
                  background: 'var(--bg-elevated)',
                  border: '1px solid var(--border-dim)',
                  fontSize: 10, fontFamily: 'var(--text-mono)', color: 'var(--text-secondary)',
                }}>
                  Frame {frame.frame_idx}
                </div>
              )}
            </div>
          )}

          {/* ── BOTTOM STATUS BAR ────────────────────────────── */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 20, padding: '6px 16px',
            borderTop: '1px solid var(--border-dim)', background: 'var(--bg-surface)',
            flexShrink: 0,
          }}>
            <span style={{ fontSize: 10, fontFamily: 'var(--text-mono)', color: 'var(--text-dim)' }}>
              {metrics ? `${metrics.latency_ms.total.toFixed(0)}ms pipeline` : 'Awaiting data…'}
            </span>
            {metrics && (
              <>
                <span style={{ fontSize: 10, fontFamily: 'var(--text-mono)', color: 'var(--green)' }}>
                  {metrics.memory_saved_pct.toFixed(1)}% memory saved vs uniform
                </span>
                <span style={{ fontSize: 10, fontFamily: 'var(--text-mono)', color: 'var(--text-dim)' }}>
                  {metrics.point_count.toLocaleString()} pts/frame
                </span>
              </>
            )}
          </div>
        </div>

        {/* ── RIGHT SIDEBAR ───────────────────────────────── */}
        <div style={{
          width: 240, display: 'flex', flexDirection: 'column',
          borderLeft: '1px solid var(--border-dim)',
          background: 'var(--bg-surface)', overflow: 'hidden', flexShrink: 0,
        }}>
          {/* Metrics */}
          <div style={{ flex: 1, overflowY: 'auto', padding: 16 }}>
            <div style={{ fontSize: 10, color: 'var(--text-dim)', letterSpacing: '0.1em',
              textTransform: 'uppercase', marginBottom: 12 }}>
              Live Telemetry
            </div>
            <MetricsPanel metrics={metrics} />
          </div>

          {/* Divider */}
          <div style={{ height: 1, background: 'var(--border-dim)' }} />

          {/* Legend */}
          <div style={{ padding: 16, flexShrink: 0 }}>
            <Legend />
          </div>
        </div>
      </div>
    </div>
  )
}
