// ============================================================
// apiClient.ts — Typed fetch wrappers for PRAHARI backend
// ============================================================
import { API_BASE } from './constants'
import type { FramePayload, MetricsPayload } from './types'

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`)
  if (!res.ok) throw new Error(`HTTP ${res.status} ${path}`)
  return res.json() as Promise<T>
}

async function post<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { method: 'POST' })
  if (!res.ok) throw new Error(`HTTP ${res.status} ${path}`)
  return res.json() as Promise<T>
}

export const api = {
  health: ()              => get<{ status: string }>('/health'),
  frame:  ()              => get<FramePayload>('/frame'),
  frameUniform: ()        => get<FramePayload>('/frame/uniform'),
  frameByIdx: (i: number) => get<FramePayload>(`/frame/${i}`),
  metrics: ()             => get<MetricsPayload>('/metrics'),

  playbackStart: ()       => post('/playback/start'),
  playbackStop:  ()       => post('/playback/stop'),
  playbackSpeed: (x: number) => post(`/playback/speed?x=${x}`),
  playbackSeek:  (f: number) => post(`/playback/seek?frame=${f}`),

  switchBackend: (name: 'groundtruth' | 'model') =>
    post(`/perception/backend?name=${name}`),
}
