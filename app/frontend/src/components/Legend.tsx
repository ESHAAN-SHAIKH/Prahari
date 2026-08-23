// ============================================================
// Legend.tsx — Class colour legend, imports from constants (single source)
// ============================================================
import { CLASS_COLORS, CLASS_NAMES, TIER_COLORS, TIER_LABELS } from '../lib/constants'

export default function Legend() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Semantic classes */}
      <div>
        <div style={{ fontSize: 10, color: 'var(--text-dim)', letterSpacing: '0.1em',
          textTransform: 'uppercase', marginBottom: 8 }}>
          Semantic Classes
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {Object.entries(CLASS_NAMES).map(([id, name]) => (
            <div key={id} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div style={{
                width: 10, height: 10, borderRadius: 2, flexShrink: 0,
                background: CLASS_COLORS[Number(id)],
                boxShadow: `0 0 6px ${CLASS_COLORS[Number(id)]}80`,
              }} />
              <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{name}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Tier rings */}
      <div>
        <div style={{ fontSize: 10, color: 'var(--text-dim)', letterSpacing: '0.1em',
          textTransform: 'uppercase', marginBottom: 8 }}>
          Resolution Tiers
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {[0, 1, 2].map(tier => (
            <div key={tier} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div style={{
                width: 10, height: 2, borderRadius: 1, flexShrink: 0,
                background: TIER_COLORS[tier],
              }} />
              <span style={{ fontSize: 11, color: 'var(--text-dim)', fontFamily: 'var(--text-mono)' }}>
                {TIER_LABELS[tier]}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
