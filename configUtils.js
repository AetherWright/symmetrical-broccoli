export function readNumberEnv(name, fallback, { min = -Infinity, max = Infinity } = {}) {
  const raw = process.env?.[name]
  const parsed = raw != null ? Number(raw) : NaN
  if (Number.isFinite(parsed)) {
    if (parsed < min) return min
    if (parsed > max) return max
    return parsed
  }
  return fallback
}
