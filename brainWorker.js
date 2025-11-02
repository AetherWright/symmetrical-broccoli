import { parentPort } from 'node:worker_threads'

import {
  chooseAction,
  trainBrain,
  chooseActionsBatch,
  trainBrainsBatch,
  getRemoteStateSnapshot
} from './brainClient.js'

if (!parentPort) {
  throw new Error('brainWorker must be started as a worker thread')
}

const handlers = {
  chooseAction: {
    single: chooseAction,
    batch: chooseActionsBatch
  },
  trainBrain: {
    single: trainBrain,
    batch: trainBrainsBatch
  }
}

function serializeError(error) {
  if (!error) {
    return { message: 'Unknown error', name: 'Error' }
  }
  return {
    message: error.message ?? String(error ?? 'Error'),
    name: error.name ?? 'Error',
    stack: error.stack ?? null,
    code: error.code ?? null,
    retryAt: error.retryAt ?? null
  }
}

parentPort.on('message', async message => {
  const { id, op, args = [] } = message ?? {}
  if (typeof id === 'undefined') {
    return
  }
  const handler = handlers[op]
  if (!handler) {
    parentPort.postMessage({
      id,
      ok: false,
      error: serializeError(new Error(`Unsupported operation: ${op}`)),
      remoteState: getRemoteStateSnapshot()
    })
    return
  }

  try {
    const batch = Array.isArray(message.batch) ? message.batch : null
    if (batch && batch.length > 0 && handler.batch) {
      const batchArgs = batch.map(entry => entry.args ?? [])
      const batchResults = await handler.batch(batchArgs)
      const results = batch.map((entry, index) => {
        const current = batchResults?.[index]
        if (!current) {
          return { id: entry.id, ok: false, error: serializeError(new Error('Batch entry missing result')) }
        }
        if (current.ok) {
          return { id: entry.id, ok: true, result: current.value }
        }
        return {
          id: entry.id,
          ok: false,
          error: serializeError(current.error ?? new Error('Batch entry failed'))
        }
      })
      parentPort.postMessage({
        id,
        ok: true,
        results,
        remoteState: getRemoteStateSnapshot()
      })
    } else {
      const effectiveArgs = batch && batch.length === 1 ? batch[0].args ?? [] : args
      const result = await handler.single(...effectiveArgs)
      parentPort.postMessage({
        id,
        ok: true,
        results: [
          {
            id: batch && batch.length === 1 ? batch[0].id : id,
            ok: true,
            result
          }
        ],
        remoteState: getRemoteStateSnapshot()
      })
    }
  } catch (error) {
    parentPort.postMessage({
      id,
      ok: false,
      error: serializeError(error),
      remoteState: getRemoteStateSnapshot()
    })
  }
})
