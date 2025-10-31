import process from 'node:process'

const API_BASE_URL = process.env.TF_SERVER_URL?.replace(/\/$/, '') ?? 'http://127.0.0.1:5000'

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
  const response = await fetchImpl(url, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined
  })

  if (!response.ok) {
    if (allowNotFound && response.status === 404) {
      return null
    }
    const text = await response.text().catch(() => '')
    throw new Error(`Remote brain request failed: ${response.status} ${response.statusText} -> ${text}`)
  }

  if (response.status === 204) {
    return null
  }

  return response.json()
}

function ensureBrainId(brain) {
  if (!brain || !brain.id) {
    throw new Error('Brain reference missing identifier')
  }
  return brain.id
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
    observation,
    epsilon,
    bot_id: brain.owner ?? null
  }
  const result = await request(`/api/brains/${brainId}/act`, { body: payload })
  if (!Number.isInteger(result?.action)) {
    throw new Error('Remote brain did not return a valid action index')
  }
  return result.action
}

export async function trainBrain(brain, observation, actionIndex, reward, nextObservation) {
  const brainId = ensureBrainId(brain)
  const payload = {
    observation,
    action: actionIndex,
    reward,
    next_observation: nextObservation,
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

