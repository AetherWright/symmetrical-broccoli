const DEFAULT_INFER_URL = (process.env.ECHO_INFER_URL || 'http://localhost:8000/infer').replace(/\/$/, '')
const DEFAULT_LEARN_URL = (process.env.ECHO_LEARN_URL || 'http://localhost:8000/learn').replace(/\/$/, '')

function parseBool(value, fallback = true) {
  if (value == null) return fallback
  const normalized = String(value).trim().toLowerCase()
  if (['1', 'true', 'yes', 'on'].includes(normalized)) return true
  if (['0', 'false', 'no', 'off'].includes(normalized)) return false
  return fallback
}

function parseNumber(value, fallback) {
  if (value == null || value === '') return fallback
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

const config = {
  inferUrl: DEFAULT_INFER_URL,
  learnUrl: DEFAULT_LEARN_URL,
  timeoutMs: Math.max(100, Math.floor(parseNumber(process.env.ECHO_TIMEOUT, 8) * 1000)),
  maxRetries: Math.max(1, Math.floor(parseNumber(process.env.ECHO_MAX_RETRIES, 3))),
  retryBackoffMs: Math.max(50, Math.floor(parseNumber(process.env.ECHO_RETRY_BACKOFF, 0.5) * 1000)),
  preferPipeline: parseBool(process.env.ECHO_PREFER_PIPELINE, true),
  defaultTemperature: parseNumber(process.env.ECHO_TEMPERATURE, 0.7)
}

const fallbackEnabled = parseBool(process.env.ECHO_FALLBACK_ENABLED, true)

function ensureFetch() {
  if (typeof fetch !== 'function') {
    throw new Error('Global fetch API is required for Echo fallback requests')
  }
  return fetch
}

function toPlainArray(value) {
  if (value == null) return value
  if (Array.isArray(value)) return value
  if (ArrayBuffer.isView(value) && typeof value.length === 'number') {
    return Array.from(value)
  }
  return value
}

function sanitizeObservation(observation) {
  const plain = toPlainArray(observation)
  if (!Array.isArray(plain)) {
    return {
      values: plain,
      meta: { replaced: 0, clipped: 0, adjusted: false }
    }
  }
  const sanitized = new Array(plain.length)
  let replaced = 0
  for (let i = 0; i < plain.length; i++) {
    const num = Number(plain[i])
    if (!Number.isFinite(num)) {
      sanitized[i] = 0
      replaced += 1
    } else {
      sanitized[i] = num
    }
  }
  return {
    values: sanitized,
    meta: { replaced, clipped: 0, adjusted: false }
  }
}

function computeTemperature(epsilon = 0.1) {
  if (!Number.isFinite(epsilon)) {
    return config.defaultTemperature
  }
  const normalized = Math.max(0, Math.min(1, epsilon))
  const adjusted = config.defaultTemperature + (normalized - 0.1) * 1.5
  return Math.max(0.05, Math.min(2.5, adjusted))
}

async function sleep(ms) {
  await new Promise(resolve => setTimeout(resolve, ms))
}

async function postWithRetries(url, payload) {
  const fetchImpl = ensureFetch()
  let attempt = 0
  let delay = config.retryBackoffMs
  let lastError

  while (attempt < config.maxRetries) {
    attempt += 1
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), config.timeoutMs)
    try {
      const response = await fetchImpl(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal
      })
      clearTimeout(timer)
      if (!response.ok) {
        const text = await response.text().catch(() => '')
        throw new Error(`Echo request failed (${response.status} ${response.statusText}): ${text}`)
      }
      const data = await response.json().catch(() => ({}))
      return data
    } catch (error) {
      clearTimeout(timer)
      lastError = error
      if (attempt >= config.maxRetries) {
        break
      }
      await sleep(delay)
      delay *= 2
    }
  }

  throw lastError || new Error(`Echo request to ${url} failed`)
}

function mapActionToIndex(actionValue, actions) {
  if (!Array.isArray(actions) || !actions.length) {
    if (Number.isInteger(actionValue)) {
      return Math.max(0, actionValue)
    }
    return 0
  }

  if (typeof actionValue === 'string') {
    const index = actions.indexOf(actionValue)
    if (index >= 0) {
      return index
    }
  }

  if (Number.isFinite(actionValue)) {
    const idx = Math.round(actionValue)
    if (idx >= 0 && idx < actions.length) {
      return idx
    }
  }

  return 0
}

function actionLabelFromIndex(index, actions) {
  if (!Array.isArray(actions) || !actions.length) {
    return null
  }
  if (!Number.isInteger(index) || index < 0 || index >= actions.length) {
    return actions[0]
  }
  return actions[index]
}

export function isEchoFallbackEnabled() {
  return fallbackEnabled && Boolean(config.inferUrl) && Boolean(config.learnUrl)
}

export async function chooseEchoAction({
  observation,
  epsilon = 0.1,
  actions = [],
  botId = null
} = {}) {
  if (!isEchoFallbackEnabled()) {
    throw new Error('Echo fallback is not enabled')
  }

  const sanitized = sanitizeObservation(observation)
  const payload = {
    observations: sanitized.values,
    temperature: computeTemperature(epsilon),
    prefer_pipeline: config.preferPipeline
  }
  if (!Array.isArray(payload.observations)) {
    payload.observations = sanitized.values
  }
  if (Array.isArray(actions) && actions.length) {
    payload.actions = actions.map(action => (typeof action === 'string' ? action : String(action)))
  }
  if (botId) {
    payload.bot_id = botId
  }

  const response = await postWithRetries(config.inferUrl, payload)
  const actionValue = response?.action
  const confidenceRaw = response?.confidence
  const confidence = Number.isFinite(Number(confidenceRaw)) ? Number(confidenceRaw) : null

  const index = mapActionToIndex(actionValue, actions)
  const label = actionLabelFromIndex(index, actions) || (typeof actionValue === 'string' ? actionValue : null)

  return {
    action: index,
    weightsOk: true,
    confidence,
    sanitization: {
      observation: sanitized.meta,
      remote: {
        provider: 'echo',
        fallback: true,
        confidence,
        action: label,
        policy: { replaced: 0, clipped: 0, adjusted: false }
      }
    }
  }
}

export async function reportEchoLearning({
  observation,
  nextObservation,
  actionIndex,
  actions = [],
  reward = 0,
  botId = null
} = {}) {
  if (!isEchoFallbackEnabled()) {
    throw new Error('Echo fallback is not enabled')
  }

  const sanitizedObservation = sanitizeObservation(observation)
  const sanitizedNext = nextObservation != null
    ? sanitizeObservation(nextObservation)
    : { values: null, meta: { replaced: 0, clipped: 0, adjusted: false } }

  const rewardValue = Number.isFinite(Number(reward)) ? Number(reward) : 0
  const actionLabel = actionLabelFromIndex(actionIndex, actions) ?? actionIndex

  const payload = {
    observation: sanitizedObservation.values,
    action: actionLabel,
    reward: rewardValue
  }
  if (Array.isArray(sanitizedNext.values)) {
    payload.next_observation = sanitizedNext.values
  }
  if (botId) {
    payload.bot_id = botId
  }

  await postWithRetries(config.learnUrl, payload)

  return {
    trained: false,
    weightsOk: true,
    droppedGradients: 0,
    clippedGradients: 0,
    gradientNorm: null,
    learningRate: null,
    hebbian: null,
    sanitization: {
      observation: sanitizedObservation.meta,
      nextObservation: sanitizedNext.meta,
      remote: {
        provider: 'echo',
        fallback: true
      }
    }
  }
}

export function getEchoConfig() {
  return { ...config }
}
