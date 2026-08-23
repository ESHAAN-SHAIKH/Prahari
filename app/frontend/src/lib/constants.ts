// ============================================================
// constants.ts — Single source of truth for class colours,
// tier colours, and shared visual configuration.
// Import from here in ALL components — never define twice.
// ============================================================

export const CLASS_COLORS: Record<number, string> = {
  0:   '#00e5a0', // terrain_drivable  — green
  1:   '#f5a623', // static_obstacle   — amber
  2:   '#ff4757', // dynamic_pedestrian— red
  3:   '#00c8ff', // dynamic_vehicle   — cyan
  255: '#3a4a5c', // unknown           — slate
}

export const CLASS_NAMES: Record<number, string> = {
  0:   'Terrain',
  1:   'Obstacle',
  2:   'Pedestrian',
  3:   'Vehicle',
  255: 'Unknown',
}

export const TIER_CELL_SIZES: Record<number, number> = {
  0: 0.05,
  1: 0.15,
  2: 0.50,
}

export const TIER_LABELS: Record<number, string> = {
  0: 'Near  0–10m  5cm',
  1: 'Mid  10–30m  15cm',
  2: 'Far  30–100m  50cm',
}

export const TIER_COLORS: string[] = ['#00c8ff', '#7a6fff', '#ff7a6f']

export const API_BASE = 'http://localhost:8000'
export const POLL_INTERVAL_MS = 300
export const TARGET_FPS = 10
