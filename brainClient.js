import process from 'node:process'

const API_BASE_URL = process.env.TF_SERVER_URL?.replace(/\/$/, '') ?? 'http://127.0.0.1:5000'

const MIN_RETRY_DELAY_MS = 1000
const MAX_RETRY_DELAY_MS = 30000

export class RemoteBrainUnavailableError extends Error {
  constructor(message, { retryAt = null, cause = null } = {}) {
    super(message)
    this.name = 'RemoteBrainUnavailableError'
    this.code = 'REMOTE_BRAIN_UNAVAILABLE'
    this.retryAt = retryAt
    if (cause) {
      this.cause = cause
    }
  }
}

const remoteState = {
  connected: true,
  failureCount: 0,
  retryDelay: MIN_RETRY_DELAY_MS,
  blockUntil: 0,
  lastError: null,
  lastFailureAt: 0
}

function computeNextDelay(delay) {
  const next = delay * 2
  return Math.min(MAX_RETRY_DELAY_MS, Math.max(MIN_RETRY_DELAY_MS, next))
}

function markRemoteConnected() {
  if (!remoteState.connected) {
    remoteState.failureCount = 0
    remoteState.retryDelay = MIN_RETRY_DELAY_MS
    remoteState.blockUntil = 0
    remoteState.lastError = null
    remoteState.lastFailureAt = 0
  }
  remoteState.connected = true
}

function markRemoteFailure(error, { retryAfterMs } = {}) {
  const now = Date.now()
  remoteState.connected = false
  remoteState.failureCount += 1
  remoteState.lastError = error?.message ?? String(error ?? 'Remote brain request failed')
  remoteState.lastFailureAt = now
  const delay = typeof retryAfterMs === 'number' && retryAfterMs > 0
    ? Math.min(MAX_RETRY_DELAY_MS, Math.max(MIN_RETRY_DELAY_MS, retryAfterMs))
    : computeNextDelay(remoteState.retryDelay)
  remoteState.retryDelay = delay
  remoteState.blockUntil = now + delay
}

export function isRemoteBrainConnected() {
  if (!remoteState.connected && Date.now() >= remoteState.blockUntil) {
    remoteState.connected = true
  }
  return remoteState.connected
}

export function getRemoteBrainStatus() {
  const now = Date.now()
  const connected = isRemoteBrainConnected()
  return {
    connected,
    failureCount: remoteState.failureCount,
    retryDelay: remoteState.retryDelay,
    retryAt: connected ? null : remoteState.blockUntil,
    lastError: remoteState.lastError,
    lastFailureAt: remoteState.lastFailureAt,
    canProbe: now >= remoteState.blockUntil
  }
}

function assertFetch() {
  if (typeof fetch !== 'function') {
    throw new Error('Global fetch API is not available in this Node.js runtime')
  }
  return fetch
}

async function request(path, { method = 'POST', body, allowNotFound = false } = {}) {
  const fetchImpl = assertFetch()
  const url = `${API_BASE_URL}${path}`
  const headers = { 'Content-Type': 'application/json' }
  const status = getRemoteBrainStatus()

  if (!status.connected && !status.canProbe) {
    throw new RemoteBrainUnavailableError(
      status.lastError ?? 'Remote brain API unavailable',
      { retryAt: status.retryAt }
    )
  }

  let response
  try {
    response = await fetchImpl(url, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined
    })
  } catch (err) {
    markRemoteFailure(err)
    throw new RemoteBrainUnavailableError('Failed to reach remote brain service', {
      retryAt: remoteState.blockUntil,
      cause: err
    })
  }

  if (!response.ok) {
    if (response.status === 503) {
      const retryAfterHeader = response.headers?.get?.('retry-after')
      const retryAfterSeconds = retryAfterHeader ? Number.parseFloat(retryAfterHeader) : NaN
      const retryAfterMs = Number.isFinite(retryAfterSeconds) ? retryAfterSeconds * 1000 : undefined
      markRemoteFailure(new Error('Remote brain service returned 503'), { retryAfterMs })
      throw new RemoteBrainUnavailableError('Remote brain service unavailable', {
        retryAt: remoteState.blockUntil
      })
    }

    if (allowNotFound && response.status === 404) {
      markRemoteConnected()
      return null
    }

    const text = await response.text().catch(() => '')
    throw new Error(`Remote brain request failed: ${response.status} ${response.statusText} -> ${text}`)
  }

  if (response.status === 204) {
    markRemoteConnected()
    return null
  }

  let data
  try {
    data = await response.json()
  } catch (err) {
    markRemoteFailure(err)
    throw err
  }

  markRemoteConnected()
  return data
}

function ensureBrainId(brain) {
  if (!brain || !brain.id) {
    throw new Error('Brain reference missing identifier')
  }
  return brain.id
}

function toPlainList(observation) {
  if (observation == null) return observation
  if (Array.isArray(observation)) return observation
  if (ArrayBuffer.isView(observation) && typeof observation.length === 'number') {
    return Array.from(observation)
  }
  if (typeof observation === 'object' && typeof observation[Symbol.iterator] === 'function') {
    return Array.from(observation)
  }
  return observation
}

export async function createBrain(inputSize, actionCount) {
  const payload = { input_size: inputSize, action_count: actionCount }
  const result = await request('/api/brains', { body: payload })
  return {
    id: result?.brain_id,
    inputSize,
    actionCount,
    owner: null
  }
}

export async function chooseAction(brain, observation, epsilon = 0.1) {
  const brainId = ensureBrainId(brain)
  const payload = {
    observation: toPlainList(observation),
    epsilon,
    bot_id: brain.owner ?? null
  }
  const result = await request(`/api/brains/${brainId}/act`, { body: payload })
  if (!Number.isInteger(result?.action)) {
    throw new Error('Remote brain did not return a valid action index')
  }
  const prediction = Array.isArray(result?.prediction) ? result.prediction : null
  return { action: result.action, prediction }
}

export async function trainBrain(brain, observation, actionIndex, reward, nextObservation) {
  const brainId = ensureBrainId(brain)
  const payload = {
    observation: toPlainList(observation),
    action: actionIndex,
    reward,
    next_observation: toPlainList(nextObservation),
    bot_id: brain.owner ?? null
  }
  const result = await request(`/api/brains/${brainId}/train`, { body: payload })
  return Boolean(result?.trained)
}

export async function saveBrain(brain, dir) {
  if (!brain) return
  const brainId = ensureBrainId(brain)
  await request(`/api/brains/${brainId}/save`, { body: { path: dir } })
}

export async function loadBrain(dir, inputSize, actionCount) {
  const payload = { path: dir, input_size: inputSize, action_count: actionCount }
  const result = await request('/api/brains/load', { body: payload, allowNotFound: true })
  if (!result) return null
  return {
    id: result?.brain_id,
    inputSize,
    actionCount,
    owner: null
  }
}

export const DEFAULT_BRAIN_DIR = 'tf_server_brain_checkpoint'

export async function saveBrainState(state, dir = DEFAULT_BRAIN_DIR) {
  await request('/api/state/save', { body: { path: dir, state } })
}

export async function loadBrainState(dir = DEFAULT_BRAIN_DIR) {
  const result = await request('/api/state/load', { body: { path: dir } })
  return result?.state ?? null
}

export async function copyWeights(target, source) {
  if (!target || !source) return
  const targetId = ensureBrainId(target)
  const sourceId = ensureBrainId(source)
  await request(`/api/brains/${targetId}/copy`, { body: { source_id: sourceId } })
}

export async function averageWeights(target, models = []) {
  if (!target) return
  const ids = models.map(model => model?.id).filter(Boolean)
  if (!ids.length) return
  const targetId = ensureBrainId(target)
  await request(`/api/brains/${targetId}/average`, { body: { source_ids: ids } })
}

export async function mutateWeights(brain, stddev = 0.02) {
  if (!brain) return
  const brainId = ensureBrainId(brain)
  await request(`/api/brains/${brainId}/mutate`, { body: { stddev } })
}
