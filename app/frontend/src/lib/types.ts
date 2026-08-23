// ============================================================
// types.ts — Shared TypeScript interfaces for API responses
// ============================================================

export interface GridCell {
  x: number
  y: number
  size: number
  height: number
  class_id: number
}

export interface FramePayload {
  frame_idx: number | null
  cell_count: number
  cells: GridCell[]
}

export interface LatencyBreakdown {
  classify: number
  project: number
  serialize: number
  total: number
}

export interface MetricsPayload {
  fps: number
  latency_ms: LatencyBreakdown
  memory_saved_pct: number
  point_count: number
  backend: 'groundtruth' | 'model'
  window_samples: number
}

export interface PlaybackStatus {
  is_playing: boolean
  current_frame: number
  speed_multiplier: number
  total_frames: number
}

export type ViewMode = 'adaptive' | 'uniform' | 'comparison'
