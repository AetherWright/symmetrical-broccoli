import { parentPort } from 'node:worker_threads'

import { chooseAction, trainBrain, getRemoteStateSnapshot } from './brainClient.js'

if (!parentPort) {
  throw new Error('brainWorker must be started as a worker thread')
}

const handlers = {
  chooseAction,
  trainBrain
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
  const fn = handlers[op]
  if (typeof fn !== 'function') {
    parentPort.postMessage({
      id,
      ok: false,
      error: serializeError(new Error(`Unsupported operation: ${op}`)),
      remoteState: getRemoteStateSnapshot()
    })
    return
  }

  try {
    const result = await fn(...args)
    parentPort.postMessage({
      id,
      ok: true,
      result,
      remoteState: getRemoteStateSnapshot()
    })
  } catch (error) {
    parentPort.postMessage({
      id,
      ok: false,
      error: serializeError(error),
      remoteState: getRemoteStateSnapshot()
    })
  }
})
