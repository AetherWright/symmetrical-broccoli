import { performance } from 'node:perf_hooks'

export function monotonicNow() {
  if (typeof performance?.now === 'function') {
    return performance.now()
  }
  return Date.now()
}
