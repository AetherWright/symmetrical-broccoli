import os from 'node:os'
import { Worker } from 'node:worker_threads'

import {
  RemoteBrainUnavailableError,
  chooseAction as directChooseAction,
  trainBrain as directTrainBrain,
  applyRemoteStateSnapshot
} from './brainClient.js'

const WORKER_MODULE = new URL('./brainWorker.js', import.meta.url)

class BrainWorkerPool {
  constructor(size) {
    this.size = Math.max(1, size | 0)
    this.workers = []
    this.queue = []
    this.nextJobId = 1
    this._destroyed = false
    this.batchSize = Math.max(1, parseBatchSize())
    for (let i = 0; i < this.size; i += 1) {
      this._spawnWorker()
    }
  }

  get destroyed() {
    return this._destroyed
  }

  _spawnWorker() {
    if (this._destroyed) return
    const worker = new Worker(WORKER_MODULE, { type: 'module' })
    const wrapper = {
      worker,
      free: true,
      currentJob: null
    }
    worker.on('message', message => {
      this._handleMessage(wrapper, message)
    })
    worker.on('error', error => {
      this._handleWorkerFailure(wrapper, error)
    })
    worker.on('exit', code => {
      this._handleWorkerExit(wrapper, code)
    })
    this.workers.push(wrapper)
  }

  async destroy() {
    if (this._destroyed) return
    this._destroyed = true
    const pendingError = new Error('Brain worker pool shut down')
    while (this.queue.length) {
      const job = this.queue.shift()
      job.reject(pendingError)
    }
    const terminations = this.workers.map(wrapper => {
      const { currentJob } = wrapper
      if (currentJob) {
        for (const job of currentJob.jobs ?? []) {
          job.reject(pendingError)
        }
      }
      return wrapper.worker.terminate()
    })
    this.workers.length = 0
    await Promise.allSettled(terminations)
  }

  run(op, args) {
    if (this._destroyed) {
      return Promise.reject(new Error('Brain worker pool is destroyed'))
    }
    return new Promise((resolve, reject) => {
      const job = {
        id: this.nextJobId++,
        op,
        args,
        resolve,
        reject
      }
      this.queue.push(job)
      this._dispatch()
    })
  }

  _dispatch() {
    for (const wrapper of this.workers) {
      if (!this.queue.length) break
      if (!wrapper.free || wrapper.currentJob) continue
      const job = this.queue.shift()
      const batch = [job]
      while (
        batch.length < this.batchSize &&
        this.queue.length &&
        this.queue[0].op === job.op
      ) {
        batch.push(this.queue.shift())
      }
      wrapper.currentJob = {
        op: job.op,
        jobs: batch,
        id: job.id
      }
      wrapper.free = false
      try {
        wrapper.worker.postMessage({
          id: job.id,
          op: job.op,
          batch: batch.map(entry => ({ id: entry.id, args: entry.args }))
        })
      } catch (error) {
        wrapper.free = true
        const failedGroup = wrapper.currentJob
        wrapper.currentJob = null
        if (failedGroup) {
          for (const item of failedGroup.jobs) {
            item.reject(error)
          }
        } else {
          job.reject(error)
        }
      }
    }
  }

  _handleMessage(wrapper, message) {
    const { id, ok, result, results, error, remoteState } = message ?? {}
    if (remoteState) {
      applyRemoteStateSnapshot(remoteState)
    }
    const current = wrapper.currentJob
    if (!current || current.id !== id) {
      wrapper.currentJob = null
      wrapper.free = true
      this._dispatch()
      return
    }
    wrapper.currentJob = null
    wrapper.free = true
    if (ok && Array.isArray(results) && results.length) {
      const lookup = new Map(current.jobs.map(item => [item.id, item]))
      for (const entry of results) {
        const job = lookup.get(entry.id)
        if (!job) {
          continue
        }
        if (entry.ok) {
          job.resolve(entry.result)
        } else {
          job.reject(deserializeError(entry.error))
        }
        lookup.delete(entry.id)
      }
      for (const remaining of lookup.values()) {
        remaining.reject(new Error('Brain worker batch response missing result'))
      }
    } else if (current.jobs.length === 1) {
      const [job] = current.jobs
      if (ok) {
        job.resolve(result)
      } else {
        job.reject(deserializeError(error))
      }
    } else {
      for (const job of current.jobs) {
        if (ok) {
          job.resolve(result)
        } else {
          job.reject(deserializeError(error))
        }
      }
    }
    this._dispatch()
  }

  _handleWorkerFailure(wrapper, error) {
    this._rejectCurrentJob(wrapper, error)
    this._removeWorker(wrapper)
    if (!this._destroyed) {
      console.warn('[BrainWorkerPool] Worker error, respawning...', error)
      this._spawnWorker()
      this._dispatch()
    }
  }

  _handleWorkerExit(wrapper, code) {
    const exitError = code === 0 || this._destroyed
      ? null
      : new Error(`Brain worker exited with code ${code}`)
    if (exitError) {
      console.warn('[BrainWorkerPool] Worker exited unexpectedly, respawning...')
    }
    if (exitError) {
      this._rejectCurrentJob(wrapper, exitError)
    } else {
      this._rejectCurrentJob(wrapper)
    }
    this._removeWorker(wrapper)
    if (!this._destroyed && this.workers.length < this.size) {
      this._spawnWorker()
      this._dispatch()
    }
  }

  _rejectCurrentJob(wrapper, error) {
    const current = wrapper.currentJob
    wrapper.currentJob = null
    wrapper.free = true
    if (!current) {
      return
    }
    for (const job of current.jobs ?? []) {
      if (error) {
        job.reject(deserializeError(error))
      } else {
        job.reject(new Error('Brain worker terminated before completing job'))
      }
    }
  }

  _removeWorker(wrapper) {
    const index = this.workers.indexOf(wrapper)
    if (index >= 0) {
      this.workers.splice(index, 1)
    }
  }
}

function parseBatchSize() {
  const configured = process.env.BRAIN_BATCH_SIZE
  if (configured != null) {
    const parsed = Number.parseInt(configured, 10)
    if (Number.isFinite(parsed) && parsed > 0) {
      return parsed
    }
  }
  return MAX_BATCH_SIZE_FALLBACK
}

const MAX_BATCH_SIZE_FALLBACK = 16

function deserializeError(payload) {
  if (!payload) {
    return new Error('Brain worker returned an unknown error')
  }
  if (payload instanceof Error) {
    return payload
  }
  if (payload.code === 'REMOTE_BRAIN_UNAVAILABLE') {
    const err = new RemoteBrainUnavailableError(payload.message ?? 'Remote brain unavailable', {
      retryAt: payload.retryAt ?? null
    })
    if (payload.stack) {
      err.stack = payload.stack
    }
    return err
  }
  const error = new Error(payload.message ?? 'Brain worker error')
  error.name = payload.name ?? 'Error'
  if (payload.stack) {
    error.stack = payload.stack
  }
  if (payload.code) {
    error.code = payload.code
  }
  if (payload.retryAt) {
    error.retryAt = payload.retryAt
  }
  return error
}

function parseWorkerCount() {
  const configured = process.env.BRAIN_WORKER_THREADS
  if (configured != null) {
    const parsed = Number.parseInt(configured, 10)
    if (Number.isFinite(parsed) && parsed >= 0) {
      return parsed
    }
  }
  const cpuCount = Math.max(1, os.cpus()?.length ?? 1)
  const configuredBots = Number.parseInt(process.env.BOT_COUNT ?? '', 10)
  const botCount = Number.isFinite(configuredBots) && configuredBots > 0 ? configuredBots : 1
  const desired = Math.max(2, Math.ceil(botCount / 2))
  return Math.min(cpuCount, desired)
}

let poolInstance

function ensurePool() {
  if (poolInstance === null) {
    return null
  }
  if (poolInstance === undefined) {
    const workerCount = parseWorkerCount()
    if (workerCount <= 1) {
      poolInstance = null
      return null
    }
    try {
      poolInstance = new BrainWorkerPool(workerCount)
    } catch (error) {
      console.warn('[BrainWorkerPool] Failed to start worker pool, falling back to inline calls.', error)
      poolInstance = null
    }
  }
  return poolInstance
}

function fallbackOnFailure(error) {
  if (!error) return false
  if (error instanceof RemoteBrainUnavailableError) {
    return false
  }
  if (error.name === 'Error' && error.message === 'Brain worker pool is destroyed') {
    return true
  }
  if (error.code === 'ERR_WORKER_NOT_RUNNING') {
    return true
  }
  if (error.message && /Brain worker/i.test(error.message)) {
    return true
  }
  return false
}

export function isBrainWorkerPoolActive() {
  return Boolean(poolInstance && !poolInstance.destroyed)
}

export async function chooseActionConcurrent(brain, observation, epsilon) {
  const pool = ensurePool()
  if (!pool) {
    return directChooseAction(brain, observation, epsilon)
  }
  try {
    return await pool.run('chooseAction', [brain, observation, epsilon])
  } catch (error) {
    if (fallbackOnFailure(error)) {
      await shutdownBrainWorkerPool()
      return directChooseAction(brain, observation, epsilon)
    }
    throw error
  }
}

export async function trainBrainConcurrent(brain, observation, actionIndex, reward, penalty, nextObservation) {
  const pool = ensurePool()
  if (!pool) {
    return directTrainBrain(brain, observation, actionIndex, reward, penalty, nextObservation)
  }
  try {
    return await pool.run('trainBrain', [brain, observation, actionIndex, reward, penalty, nextObservation])
  } catch (error) {
    if (fallbackOnFailure(error)) {
      await shutdownBrainWorkerPool()
      return directTrainBrain(brain, observation, actionIndex, reward, penalty, nextObservation)
    }
    throw error
  }
}

export async function shutdownBrainWorkerPool() {
  if (poolInstance && !poolInstance.destroyed) {
    await poolInstance.destroy()
  }
  poolInstance = undefined
}

export function warmBrainWorkerPool() {
  const pool = ensurePool()
  return {
    enabled: Boolean(pool),
    size: pool?.size ?? 0
  }
}
