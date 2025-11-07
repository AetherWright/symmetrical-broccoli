import process from 'node:process'
import { TransformStream as NodeTransformStream } from 'node:stream/web'

const API_BASE_URL = process.env.TF_SERVER_URL?.replace(/\/$/, '') ?? 'http://127.0.0.1:5000'

const MIN_RETRY_DELAY_MS = 1000
const MAX_RETRY_DELAY_MS = 30000
const OBSERVATION_CLAMP = Number.isFinite(Number.parseFloat(process.env.OBSERVATION_CLAMP ?? ''))
  ? Number.parseFloat(process.env.OBSERVATION_CLAMP)
  : 1e6

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

export function getRemoteStateSnapshot() {
  return {
    connected: Boolean(remoteState.connected),
    failureCount: Number(remoteState.failureCount) || 0,
    retryDelay: Number(remoteState.retryDelay) || MIN_RETRY_DELAY_MS,
    blockUntil: Number(remoteState.blockUntil) || 0,
    lastError: remoteState.lastError ?? null,
    lastFailureAt: Number(remoteState.lastFailureAt) || 0
  }
}

export function applyRemoteStateSnapshot(snapshot) {
  if (!snapshot || typeof snapshot !== 'object') {
    return
  }
  if (typeof snapshot.connected === 'boolean') {
    remoteState.connected = snapshot.connected
  }
  if (Number.isFinite(snapshot.failureCount)) {
    remoteState.failureCount = snapshot.failureCount
  }
  if (Number.isFinite(snapshot.retryDelay)) {
    remoteState.retryDelay = Math.max(MIN_RETRY_DELAY_MS, snapshot.retryDelay)
  }
  if (Number.isFinite(snapshot.blockUntil)) {
    remoteState.blockUntil = snapshot.blockUntil
  }
  if (typeof snapshot.lastError === 'string' || snapshot.lastError === null) {
    remoteState.lastError = snapshot.lastError
  }
  if (Number.isFinite(snapshot.lastFailureAt)) {
    remoteState.lastFailureAt = snapshot.lastFailureAt
  }
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

const TransformStreamCtor = typeof globalThis.TransformStream === 'function'
  ? globalThis.TransformStream
  : (typeof NodeTransformStream === 'function' ? NodeTransformStream : null)

const STREAM_SUPPORTED = typeof TransformStreamCtor === 'function'

function assertTransformStream() {
  if (!STREAM_SUPPORTED) {
    throw new Error('Web Streams API is not supported in this runtime')
  }
  return TransformStreamCtor
}

class BrainStreamConnection {
  constructor(url, fetchImpl, { onClose } = {}) {
    this.url = url
    this._fetch = fetchImpl
    this._onClose = typeof onClose === 'function' ? onClose : null
    this._writer = null
    this._reader = null
    this._connecting = null
    this._encoder = new TextEncoder()
    this._decoder = new TextDecoder()
    this._buffer = ''
    this._pending = new Map()
    this.nextId = 1
    this.closed = true
  }

  isUsable() {
    return !this.closed && this._writer != null && this._reader != null
  }

  async connect() {
    if (this.isUsable()) {
      return
    }
    if (this._connecting) {
      return this._connecting
    }
    this._connecting = this._open()
    try {
      await this._connecting
    } finally {
      this._connecting = null
    }
  }

  async _open() {
    const StreamImpl = assertTransformStream()
    const stream = new StreamImpl()
    const readable = stream.readable
    const writable = stream.writable
    this._writer = writable.getWriter()
    let response
    try {
      response = await this._fetch(this.url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/jsonl' },
        body: readable,
        duplex: 'half'
      })
    } catch (error) {
      await this._writer.abort?.(error).catch(() => {})
      this._writer = null
      throw error
    }
    if (!response?.ok) {
      const err = new Error(`Remote brain stream failed: ${response?.status ?? 0} ${response?.statusText ?? ''}`.trim())
      await this._writer.close().catch(() => {})
      this._writer = null
      throw err
    }
    if (!response.body) {
      const err = new Error('Remote brain stream missing body')
      await this._writer.close().catch(() => {})
      this._writer = null
      throw err
    }
    this._reader = response.body.getReader()
    this.closed = false
    this._buffer = ''
    this._startReader()
  }

  _startReader() {
    if (!this._reader) return
    this._readerLoop = this._readLoop()
    this._readerLoop.catch(error => {
      this._failAll(error)
    })
  }

  async _readLoop() {
    if (!this._reader) return
    try {
      while (!this.closed) {
        const { done, value } = await this._reader.read()
        if (done) break
        if (value) {
          this._buffer += this._decoder.decode(value, { stream: true })
          this._drainBuffer()
        }
      }
      if (this.closed) return
      this._buffer += this._decoder.decode(new Uint8Array(), { stream: false })
      this._drainBuffer(true)
      throw new Error('Remote brain stream closed')
    } catch (error) {
      this._failAll(error)
    }
  }

  _drainBuffer(final = false) {
    let newlineIndex = this._buffer.indexOf('\n')
    while (newlineIndex !== -1) {
      const line = this._buffer.slice(0, newlineIndex).trim()
      this._buffer = this._buffer.slice(newlineIndex + 1)
      if (line) {
        let message
        try {
          message = JSON.parse(line)
        } catch (error) {
          console.warn('[BrainStream] Failed to parse response chunk:', error)
          message = null
        }
        if (message && message.id != null) {
          const pending = this._pending.get(message.id)
          if (pending) {
            this._pending.delete(message.id)
            pending.resolve(message)
          }
        }
      }
      newlineIndex = this._buffer.indexOf('\n')
    }
    if (final && this._buffer.trim()) {
      let message
      try {
        message = JSON.parse(this._buffer.trim())
      } catch (error) {
        console.warn('[BrainStream] Failed to parse trailing response chunk:', error)
        message = null
      }
      this._buffer = ''
      if (message && message.id != null) {
        const pending = this._pending.get(message.id)
        if (pending) {
          this._pending.delete(message.id)
          pending.resolve(message)
        }
      }
    }
  }

  async send(type, payload) {
    if (!this.isUsable()) {
      throw new Error('Brain stream connection is not ready')
    }
    const id = this.nextId++
    const record = { id, type, ...payload }
    const promise = new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject })
    })
    try {
      const encoded = this._encoder.encode(`${JSON.stringify(record)}\n`)
      await this._writer.write(encoded)
    } catch (error) {
      this._pending.delete(id)
      throw error
    }
    return promise
  }

  _failAll(error) {
    if (this.closed && this._pending.size === 0) {
      return
    }
    this.closed = true
    const pendingEntries = Array.from(this._pending.values())
    this._pending.clear()
    for (const entry of pendingEntries) {
      entry.reject(error)
    }
    if (this._writer) {
      this._writer.abort?.(error).catch(() => {})
      this._writer.releaseLock?.()
      this._writer = null
    }
    if (this._reader) {
      this._reader.cancel?.(error).catch(() => {})
      this._reader.releaseLock?.()
      this._reader = null
    }
    if (typeof this._onClose === 'function') {
      this._onClose(error)
    }
  }

  destroy(error = new Error('Brain stream connection closed')) {
    this._failAll(error)
  }
}

let activeStreamConnection = null
let streamConnectPromise = null

function invalidateStreamConnection(connection, error) {
  if (!connection) return
  connection.destroy?.(error)
  if (activeStreamConnection === connection) {
    activeStreamConnection = null
  }
}

async function ensureStreamConnection() {
  if (!STREAM_SUPPORTED) {
    throw new Error('Web Streams API is unavailable')
  }
  if (activeStreamConnection && activeStreamConnection.isUsable()) {
    return activeStreamConnection
  }
  if (streamConnectPromise) {
    return streamConnectPromise
  }
  const fetchImpl = assertFetch()
  const connection = new BrainStreamConnection(
    `${API_BASE_URL}/api/brains/stream`,
    fetchImpl,
    {
      onClose: () => {
        if (activeStreamConnection === connection) {
          activeStreamConnection = null
        }
      }
    }
  )
  activeStreamConnection = connection
  streamConnectPromise = connection.connect()
    .then(() => connection)
    .catch(error => {
      invalidateStreamConnection(connection, error)
      throw error
    })
    .finally(() => {
      streamConnectPromise = null
    })
  return streamConnectPromise
}

async function sendStreamMessage(type, payload) {
  const status = getRemoteBrainStatus()
  if (!status.connected && !status.canProbe) {
    throw new RemoteBrainUnavailableError(
      status.lastError ?? 'Remote brain API unavailable',
      { retryAt: status.retryAt }
    )
  }
  let connection
  try {
    connection = await ensureStreamConnection()
  } catch (error) {
    markRemoteFailure(error)
    throw new RemoteBrainUnavailableError('Failed to establish remote brain stream', {
      retryAt: remoteState.blockUntil,
      cause: error
    })
  }
  try {
    const response = await connection.send(type, payload)
    if (response?.remote_state) {
      applyRemoteStateSnapshot(response.remote_state)
    }
    markRemoteConnected()
    return response
  } catch (error) {
    invalidateStreamConnection(connection, error)
    markRemoteFailure(error)
    throw new RemoteBrainUnavailableError('Remote brain stream unavailable', {
      retryAt: remoteState.blockUntil,
      cause: error
    })
  }
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

function sanitizeNumericValue(value, clamp = OBSERVATION_CLAMP) {
  const num = Number(value)
  if (!Number.isFinite(num)) {
    return { value: 0, replaced: true, clipped: false }
  }
  if (clamp > 0 && Math.abs(num) > clamp) {
    return {
      value: num < 0 ? -clamp : clamp,
      replaced: false,
      clipped: true
    }
  }
  return { value: num, replaced: false, clipped: false }
}

function sanitizeObservationPayload(vector, { expectedLength = null, label = 'observation' } = {}) {
  const plain = toPlainList(vector)
  if (!Array.isArray(plain)) {
    return { values: null, replaced: 0, clipped: 0, adjusted: false }
  }

  const sanitized = new Array(plain.length)
  let replaced = 0
  let clipped = 0
  for (let i = 0; i < plain.length; i++) {
    const { value, replaced: didReplace, clipped: didClip } = sanitizeNumericValue(plain[i])
    if (didReplace) replaced += 1
    if (didClip) clipped += 1
    sanitized[i] = value
  }

  let adjusted = false
  if (Number.isInteger(expectedLength) && expectedLength > 0) {
    if (sanitized.length > expectedLength) {
      sanitized.length = expectedLength
      adjusted = true
    } else if (sanitized.length < expectedLength) {
      sanitized.push(...Array(expectedLength - sanitized.length).fill(0))
      adjusted = true
    }
  }

  if (replaced > 0 || clipped > 0 || adjusted) {
    console.warn(
      `[RemoteBrain] Sanitized ${label} payload (replaced=${replaced}, clipped=${clipped}, adjusted=${adjusted})`
    )
  }

  return { values: sanitized, replaced, clipped, adjusted }
}

function summarizeSanitization(meta) {
  if (!meta) {
    return { replaced: 0, clipped: 0, adjusted: false }
  }
  return {
    replaced: meta.replaced ?? 0,
    clipped: meta.clipped ?? 0,
    adjusted: Boolean(meta.adjusted)
  }
}

function sanitizeActionIndex(action, actionCount) {
  if (!Number.isInteger(action)) {
    return { value: null, adjusted: false }
  }
  const minIndex = 0
  const maxIndex = Number.isInteger(actionCount) && actionCount > 0 ? actionCount - 1 : action
  if (action < minIndex) {
    return { value: minIndex, adjusted: true }
  }
  if (action > maxIndex) {
    return { value: maxIndex, adjusted: true }
  }
  return { value: action, adjusted: false }
}

function clampEpsilon(epsilon) {
  if (!Number.isFinite(epsilon)) {
    return 0.1
  }
  if (epsilon < 0) return 0
  if (epsilon > 0.999) return 0.999
  return epsilon
}

function sanitizeRewardComponent(value) {
  if (!Number.isFinite(value) || value <= 0) {
    return 0
  }
  const limit = 1e6
  if (value > limit) return limit
  return value
}

function makeDefaultTrainResult(sanitizedObservation, sanitizedNext) {
  return {
    trained: false,
    weightsOk: true,
    droppedGradients: 0,
    clippedGradients: 0,
    sanitization: {
      observation: summarizeSanitization(sanitizedObservation),
      nextObservation: summarizeSanitization(sanitizedNext),
      remote: {}
    }
  }
}

async function legacyBatchAct(jobs, results) {
  if (!jobs.length) return
  let response
  try {
    const payloads = jobs.map(job => job.payload)
    response = await request('/api/brains/batch_act', { body: { requests: payloads } })
  } catch (error) {
    for (const job of jobs) {
      if (!results[job.index]) {
        results[job.index] = { ok: false, error }
      }
    }
    return
  }
  const remoteResults = Array.isArray(response?.results) ? response.results : []
  for (let i = 0; i < jobs.length; i++) {
    const job = jobs[i]
    const remote = remoteResults[i]
    if (!remote || remote.error) {
      const err = remote?.error
        ? new Error(remote.error)
        : new Error('Remote brain batch action missing result')
      results[job.index] = { ok: false, error: err }
      continue
    }
    if (!Number.isInteger(remote.action)) {
      results[job.index] = {
        ok: false,
        error: new Error('Remote brain did not return a valid action index')
      }
      continue
    }
    const value = {
      action: remote.action,
      weightsOk: remote.weights_ok !== false,
      sanitization: {
        observation: summarizeSanitization(job.observation),
        remote: remote.sanitized ?? {}
      }
    }
    if (remote.reward_prediction !== undefined) {
      value.rewardPrediction = remote.reward_prediction
    }
    if (remote.meta !== undefined) {
      value.meta = remote.meta
    }
    results[job.index] = { ok: true, value }
  }
}

async function legacyBatchTrain(jobs, results) {
  if (!jobs.length) return
  let response
  try {
    const payloads = jobs.map(job => job.payload)
    response = await request('/api/brains/batch_train', { body: { requests: payloads } })
  } catch (error) {
    for (const job of jobs) {
      if (!results[job.index]) {
        results[job.index] = { ok: false, error }
      }
    }
    return
  }
  const remoteResults = Array.isArray(response?.results) ? response.results : []
  for (let i = 0; i < jobs.length; i++) {
    const job = jobs[i]
    const remote = remoteResults[i]
    if (!remote || remote.error) {
      const err = remote?.error
        ? new Error(remote.error)
        : new Error('Remote brain batch train missing result')
      results[job.index] = { ok: false, error: err }
      continue
    }
    const value = {
      trained: Boolean(remote.trained),
      weightsOk: remote.weights_ok !== false,
      droppedGradients: Number.isFinite(remote.dropped_gradients)
        ? remote.dropped_gradients
        : 0,
      clippedGradients: Number.isFinite(remote.clipped_gradients)
        ? remote.clipped_gradients
        : 0,
      gradientNorm: Number.isFinite(remote.gradient_norm)
        ? remote.gradient_norm
        : null,
      learningRate: remote.learning_rate ?? null,
      hebbian: remote.hebbian ?? null,
      sanitization: {
        observation: summarizeSanitization(job.observation),
        nextObservation: summarizeSanitization(job.nextObservation),
        remote: remote.sanitized ?? {}
      }
    }
    results[job.index] = { ok: true, value }
  }
}

export async function chooseActionsBatch(argsList) {
  const results = new Array(argsList.length)
  const jobs = []

  for (let i = 0; i < argsList.length; i++) {
    const [brain, observation, epsilon = 0.1] = argsList[i] ?? []
    const brainId = ensureBrainId(brain)
    const sanitizedObservation = sanitizeObservationPayload(observation, {
      expectedLength: brain?.inputSize,
      label: 'observation'
    })
    if (!Array.isArray(sanitizedObservation.values)) {
      results[i] = {
        ok: false,
        error: new Error('Observation must be an array-like payload')
      }
      continue
    }
    jobs.push({
      index: i,
      observation: sanitizedObservation,
      payload: {
        brain_id: brainId,
        observation: sanitizedObservation.values,
        epsilon: clampEpsilon(epsilon),
        bot_id: brain?.owner ?? null
      }
    })
  }

  if (jobs.length) {
    if (STREAM_SUPPORTED) {
      const settled = await Promise.allSettled(jobs.map(job => sendStreamMessage('act', job.payload)))
      let remoteError = null
      for (let i = 0; i < settled.length; i++) {
        const outcome = settled[i]
        const job = jobs[i]
        if (outcome.status === 'rejected') {
          remoteError = outcome.reason
          break
        }
        const message = outcome.value
        if (!message || typeof message !== 'object') {
          results[job.index] = {
            ok: false,
            error: new Error('Remote brain stream returned an invalid response')
          }
          continue
        }
        if (!message.ok) {
          const errMsg = message.error?.message ?? 'Remote brain request failed'
          results[job.index] = { ok: false, error: new Error(errMsg) }
          continue
        }
        const remote = message.result ?? {}
        if (!Number.isInteger(remote.action)) {
          results[job.index] = {
            ok: false,
            error: new Error('Remote brain did not return a valid action index')
          }
          continue
        }
        const value = {
          action: remote.action,
          weightsOk: remote.weights_ok !== false,
          sanitization: {
            observation: summarizeSanitization(job.observation),
            remote: remote.sanitized ?? {}
          }
        }
        if (remote.reward_prediction !== undefined) {
          value.rewardPrediction = remote.reward_prediction
        }
        if (remote.meta !== undefined) {
          value.meta = remote.meta
        }
        results[job.index] = { ok: true, value }
      }
      if (remoteError) {
        for (const job of jobs) {
          results[job.index] = { ok: false, error: remoteError }
        }
      }
    } else {
      await legacyBatchAct(jobs, results)
    }
  }

  for (let i = 0; i < results.length; i++) {
    if (!results[i]) {
      results[i] = {
        ok: false,
        error: new Error('Batch entry was not processed')
      }
    }
  }

  return results
}

export async function trainBrainsBatch(argsList) {
  const results = new Array(argsList.length)
  const jobs = []

  for (let i = 0; i < argsList.length; i++) {
    const [brain, observation, actionIndex, reward, penalty, nextObservation] = argsList[i] ?? []
    const brainId = ensureBrainId(brain)
    const sanitizedObservation = sanitizeObservationPayload(observation, {
      expectedLength: brain?.inputSize,
      label: 'observation'
    })
    if (!Array.isArray(sanitizedObservation.values)) {
      console.warn('[RemoteBrain] Skipping training due to invalid observation payload.')
      results[i] = {
        ok: true,
        value: makeDefaultTrainResult(sanitizedObservation, { replaced: 0, clipped: 0, adjusted: false })
      }
      continue
    }
    const sanitizedNext = nextObservation != null
      ? sanitizeObservationPayload(nextObservation, {
          expectedLength: brain?.inputSize,
          label: 'next_observation'
        })
      : { values: null, replaced: 0, clipped: 0, adjusted: false }
    const { value: sanitizedAction, adjusted: actionAdjusted } = sanitizeActionIndex(
      actionIndex,
      brain?.actionCount
    )
    if (actionIndex != null && sanitizedAction == null) {
      console.warn('[RemoteBrain] Skipping training due to invalid action index.')
      results[i] = {
        ok: true,
        value: makeDefaultTrainResult(sanitizedObservation, sanitizedNext)
      }
      continue
    }
    if (actionAdjusted) {
      console.warn('[RemoteBrain] Adjusted action index to stay within range.')
    }
    jobs.push({
      index: i,
      observation: sanitizedObservation,
      nextObservation: sanitizedNext,
      payload: {
        brain_id: brainId,
        observation: sanitizedObservation.values,
        action: sanitizedAction,
        reward: sanitizeRewardComponent(reward),
        penalty: sanitizeRewardComponent(penalty),
        next_observation: Array.isArray(sanitizedNext.values) ? sanitizedNext.values : undefined,
        bot_id: brain?.owner ?? null
      }
    })
  }

  if (jobs.length) {
    if (STREAM_SUPPORTED) {
      const settled = await Promise.allSettled(jobs.map(job => sendStreamMessage('train', job.payload)))
      let remoteError = null
      for (let i = 0; i < settled.length; i++) {
        const outcome = settled[i]
        const job = jobs[i]
        if (outcome.status === 'rejected') {
          remoteError = outcome.reason
          break
        }
        const message = outcome.value
        if (!message || typeof message !== 'object') {
          results[job.index] = {
            ok: false,
            error: new Error('Remote brain stream returned an invalid response')
          }
          continue
        }
        if (!message.ok) {
          const errMsg = message.error?.message ?? 'Remote brain request failed'
          results[job.index] = { ok: false, error: new Error(errMsg) }
          continue
        }
        const remote = message.result ?? {}
        const value = {
          trained: Boolean(remote.trained),
          weightsOk: remote.weights_ok !== false,
          droppedGradients: Number.isFinite(remote.dropped_gradients)
            ? remote.dropped_gradients
            : 0,
          clippedGradients: Number.isFinite(remote.clipped_gradients)
            ? remote.clipped_gradients
            : 0,
          gradientNorm: Number.isFinite(remote.gradient_norm)
            ? remote.gradient_norm
            : null,
          learningRate: remote.learning_rate ?? null,
          hebbian: remote.hebbian ?? null,
          sanitization: {
            observation: summarizeSanitization(job.observation),
            nextObservation: summarizeSanitization(job.nextObservation),
            remote: remote.sanitized ?? {}
          }
        }
        if (remote.reward_prediction !== undefined) {
          value.rewardPrediction = remote.reward_prediction
        }
        if (remote.meta !== undefined) {
          value.meta = remote.meta
        }
        results[job.index] = { ok: true, value }
      }
      if (remoteError) {
        for (const job of jobs) {
          results[job.index] = { ok: false, error: remoteError }
        }
      }
    } else {
      await legacyBatchTrain(jobs, results)
    }
  }

  for (let i = 0; i < results.length; i++) {
    if (!results[i]) {
      results[i] = {
        ok: false,
        error: new Error('Batch entry was not processed')
      }
    }
  }

  return results
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
  const [result] = await chooseActionsBatch([[brain, observation, epsilon]])
  if (!result || !result.ok) {
    throw (result?.error ?? new Error('Failed to select action'))
  }
  return result.value
}

export async function trainBrain(brain, observation, actionIndex, reward, penalty, nextObservation) {
  const [result] = await trainBrainsBatch([[brain, observation, actionIndex, reward, penalty, nextObservation]])
  if (!result) {
    return makeDefaultTrainResult(
      { replaced: 0, clipped: 0, adjusted: false },
      { replaced: 0, clipped: 0, adjusted: false }
    )
  }
  if (!result.ok) {
    throw result.error ?? new Error('Remote brain training failed')
  }
  return result.value
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
