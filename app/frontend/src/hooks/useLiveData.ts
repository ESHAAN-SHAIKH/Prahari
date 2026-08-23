// ============================================================
// useLiveData.ts — Polling hook driving all live data
// ============================================================
import { useEffect, useRef, useState } from 'react'
import { api } from '../lib/apiClient'
import { POLL_INTERVAL_MS } from '../lib/constants'
import type { FramePayload, MetricsPayload } from '../lib/types'

interface LiveData {
  frame:        FramePayload | null
  frameUniform: FramePayload | null
  metrics:      MetricsPayload | null
  connected:    boolean
  frameId:      number
}

export function useLiveData(isPlaying: boolean): LiveData {
  const [data, setData] = useState<LiveData>({
    frame: null, frameUniform: null, metrics: null, connected: false, frameId: 0,
  })
  const frameIdRef = useRef(0)

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>

    async function tick() {
      if (cancelled) return
      try {
        const [frame, frameUniform, metrics] = await Promise.all([
          api.frame(),
          api.frameUniform(),
          api.metrics(),
        ])
        if (!cancelled) {
          frameIdRef.current += 1
          setData({ frame, frameUniform, metrics, connected: true, frameId: frameIdRef.current })
        }
      } catch {
        if (!cancelled) setData(prev => ({ ...prev, connected: false }))
      }
      if (!cancelled) timer = setTimeout(tick, POLL_INTERVAL_MS)
    }

    tick()
    return () => { cancelled = true; clearTimeout(timer) }
  }, [isPlaying])

  return data
}
