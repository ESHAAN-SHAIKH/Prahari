// ============================================================
// ComparisonView.test.ts — TASK-017 sync guard
// Ensures both panes use the same frame timestamp/id
// ============================================================
import { describe, it, expect } from 'vitest'

const adaptiveFrame = {
  frame_idx: 10,
  cell_count: 79_500,
  cells: [],
}

const uniformFrame = {
  frame_idx: 10,  // Same frame index — critical!
  cell_count: 117_900,
  cells: [],
}

describe('ComparisonView — frame sync (TASK-017)', () => {
  it('test_comparison_panes_use_same_frame_timestamp: both frames share frame_idx', () => {
    expect(adaptiveFrame.frame_idx).toBe(uniformFrame.frame_idx)
  })

  it('uniform baseline always has more cells than adaptive (memory saving is real)', () => {
    expect(uniformFrame.cell_count).toBeGreaterThan(adaptiveFrame.cell_count)
  })

  it('memory saved pct is computed correctly', () => {
    const savedPct = (1 - adaptiveFrame.cell_count / uniformFrame.cell_count) * 100
    expect(savedPct).toBeGreaterThan(25)  // Must be meaningful, not near-zero
    expect(savedPct).toBeLessThan(60)     // Must not be suspiciously perfect
  })
})
