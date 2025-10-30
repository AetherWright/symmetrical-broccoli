// brain.js
import * as tf from '@tensorflow/tfjs'
import { promises as fs } from 'fs'
import path from 'path'

// ----------------------------
// CONFIG
// ----------------------------
export const BRAIN_CONFIG = {
  hiddenUnits: 64,
  dropoutRate: 0.2,
  useLSTM: true,   // flip this if you want temporal memory
  clipNorm: 1.0
}

export const DEFAULT_BRAIN_DIR = 'tf_brain_checkpoint'

function resolveDir(dir = DEFAULT_BRAIN_DIR) {
  return path.isAbsolute(dir) ? dir : path.resolve(process.cwd(), dir)
}

async function ensureDir(dir) {
  await fs.mkdir(dir, { recursive: true })
}

// ----------------------------
// MODEL CREATION
// ----------------------------
export function createBrain(inputSize, actionCount) {
  const model = tf.sequential()

  // Define the input layer explicitly so we can branch the architecture safely
  model.add(tf.layers.inputLayer({ inputShape: [inputSize] }))

  if (BRAIN_CONFIG.useLSTM) {
    // Reshape to include a time dimension for the LSTM
    model.add(tf.layers.reshape({ targetShape: [1, inputSize] }))
    model.add(tf.layers.lstm({
      units: BRAIN_CONFIG.hiddenUnits,
      activation: 'tanh',
      recurrentActivation: 'sigmoid',
      returnSequences: false
    }))
    // Apply normalization after the recurrent layer to stabilize training
    model.add(tf.layers.batchNormalization())
  } else {
    // Normalize the raw observations before feeding the dense stack
    model.add(tf.layers.batchNormalization())
    model.add(tf.layers.dense({
      units: BRAIN_CONFIG.hiddenUnits,
      activation: 'relu'
    }))
  }

  // Optional deeper stack
  model.add(tf.layers.dropout({ rate: BRAIN_CONFIG.dropoutRate }))
  model.add(
    tf.layers.dense({
      units: Math.max(1, Math.floor(BRAIN_CONFIG.hiddenUnits / 2)),
      activation: 'relu'
    })
  )
  model.add(tf.layers.dropout({ rate: BRAIN_CONFIG.dropoutRate }))

  // Output layer — softmax for discrete actions
  model.add(tf.layers.dense({ units: actionCount, activation: 'softmax' }))

  // Compile
  const optimizer = tf.train.rmsprop(5e-4)
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy'
  })

  console.log(`[Brain] Created model with ${BRAIN_CONFIG.hiddenUnits} hidden units`)
  return model
}

// ----------------------------
// ACTION SELECTION
// ----------------------------
export async function chooseAction(model, obs, epsilon = 0.1) {
  const probsArray = tf.tidy(() => {
    const input = tf.tensor(obs, [1, obs.length], 'float32')
    const prediction = model.predict(input)
    return prediction.dataSync()
  })

  const probs = Array.from(probsArray)

  if (Math.random() < epsilon) {
    return Math.floor(Math.random() * probs.length)
  }

  let bestIndex = 0
  let bestValue = -Infinity
  for (let i = 0; i < probs.length; i++) {
    if (probs[i] > bestValue) {
      bestValue = probs[i]
      bestIndex = i
    }
  }

  return bestIndex
}

export function copyWeights(target, source) {
  if (!target || !source) return
  const sourceWeights = source.getWeights()
  if (!sourceWeights.length) {
    sourceWeights.forEach(weight => weight.dispose?.())
    return
  }

  const clonedWeights = sourceWeights.map(weight => weight.clone())
  sourceWeights.forEach(weight => weight.dispose())

  target.setWeights(clonedWeights)
}

export function averageWeights(target, models = []) {
  if (!target || !models?.length) return
  const weightsByModel = models.map(model => model.getWeights())
  if (!weightsByModel.length) {
    return
  }

  const count = weightsByModel.length
  const weightCount = weightsByModel[0]?.length ?? 0
  if (!weightCount) {
    weightsByModel.forEach(group => group.forEach(t => t.dispose()))
    return
  }

  const averaged = []
  for (let i = 0; i < weightCount; i++) {
    const tensors = weightsByModel.map(group => group[i])
    const mean = tf.tidy(() => {
      let sum = tensors[0].clone()
      for (let j = 1; j < tensors.length; j++) {
        const next = sum.add(tensors[j])
        sum.dispose()
        sum = next
      }
      const divisor = tf.scalar(count)
      const averagedTensor = sum.div(divisor)
      sum.dispose()
      divisor.dispose()
      return averagedTensor
    })

    tensors.forEach(t => t.dispose())
    averaged.push(mean)
  }

  const clonedAverages = averaged.map(weight => weight.clone())
  averaged.forEach(weight => weight.dispose())

  target.setWeights(clonedAverages)
}

export function mutateWeights(model, stddev = 0.02) {
  if (!model || !Number.isFinite(stddev) || stddev <= 0) return
  const weights = model.getWeights()
  const mutated = []

  for (const weight of weights) {
    const next = tf.tidy(() => {
      const noise = tf.randomNormal(weight.shape, 0, stddev)
      const updated = weight.add(noise)
      noise.dispose()
      return updated
    })

    mutated.push(next)
    weight.dispose()
  }

  const clonedMutations = mutated.map(weight => weight.clone())
  mutated.forEach(weight => weight.dispose())

  model.setWeights(clonedMutations)
}

// ----------------------------
// TRAINING
// ----------------------------
export async function trainBrain(model, obs, actionIndex, reward) {
  if (typeof actionIndex !== 'number' || Number.isNaN(actionIndex)) return false

  const scaledReward = Math.max(-1, Math.min(1, reward ?? 0))

  if (!Number.isFinite(scaledReward) || scaledReward === 0) {
    return false
  }

  const outputShape = model.outputs[0].shape
  const rawUnits = outputShape ? outputShape[outputShape.length - 1] : null

  if (!Number.isFinite(rawUnits) || rawUnits <= 0) {
    return false
  }

  const outputUnits = Math.max(1, Math.floor(rawUnits))

  const boundedActionIndex = Math.max(0, Math.min(outputUnits - 1, Math.floor(actionIndex)))

  if (typeof tf.nextFrame === 'function') {
    await tf.nextFrame()
  }

  let trained = false
  tf.tidy(() => {
    const xs = tf.tensor(obs, [1, obs.length], 'float32')
    const labelBuffer = new Float32Array(outputUnits)
    labelBuffer[boundedActionIndex] = 1
    const ys = tf.tensor(labelBuffer, [1, outputUnits], 'float32')

    const minimizeFn = () => {
      const probs = model.predict(xs)
      const logProbs = probs.log()
      const selectedLogProb = logProbs.mul(ys).sum(-1)
      const scaledLoss = selectedLogProb.mul(-scaledReward)
      return scaledLoss.mean()
    }

    const lossValue = model.optimizer.minimize(minimizeFn, true)
    if (lossValue) {
      lossValue.dispose()
    }
    trained = true
  })

  return trained
}

// ----------------------------
// PERSISTENCE
// ----------------------------
export async function saveBrain(model, dir = DEFAULT_BRAIN_DIR) {
  const resolved = resolveDir(dir)
  await ensureDir(resolved)

  const handler = tf.io.withSaveHandler(async artifacts => {
    const modelJsonPath = path.join(resolved, 'model.json')
    const weightsPath = path.join(resolved, 'weights.bin')

    const weightsManifest = [{ paths: ['weights.bin'], weights: artifacts.weightSpecs ?? [] }]
    const topologyJson = JSON.stringify(artifacts.modelTopology ?? {})
    const weightsSpecsJson = JSON.stringify(artifacts.weightSpecs ?? [])
    const modelJSON = {
      modelTopology: artifacts.modelTopology ?? null,
      format: 'layers-model',
      generatedBy: 'BrainBot',
      convertedBy: null,
      trainingConfig: artifacts.trainingConfig ?? null,
      weightsManifest
    }

    await fs.writeFile(modelJsonPath, JSON.stringify(modelJSON, null, 2))
    if (artifacts.weightData) {
      const buffer = Buffer.from(artifacts.weightData)
      await fs.writeFile(weightsPath, buffer)
    }

    return {
      modelArtifactsInfo: {
        dateSaved: new Date(),
        modelTopologyType: 'JSON',
        modelTopologyBytes: Buffer.byteLength(topologyJson, 'utf8'),
        weightSpecsBytes: Buffer.byteLength(weightsSpecsJson, 'utf8'),
        weightDataBytes: artifacts.weightData ? artifacts.weightData.byteLength : 0
      }
    }
  })

  await model.save(handler)
  console.log(`[Brain] Saved model to ${resolved}`)
}

export async function loadBrain(dir = DEFAULT_BRAIN_DIR) {
  try {
    const resolved = resolveDir(dir)
    const modelJsonPath = path.join(resolved, 'model.json')
    const weightsPath = path.join(resolved, 'weights.bin')

    const handler = tf.io.withLoadHandler(async () => {
      const modelJSON = JSON.parse(await fs.readFile(modelJsonPath, 'utf8'))
      const weightBuffer = await fs.readFile(weightsPath)
      const arrayBuffer = weightBuffer.buffer.slice(
        weightBuffer.byteOffset,
        weightBuffer.byteOffset + weightBuffer.byteLength
      )

      const manifestWeights = modelJSON.weightsManifest?.[0]?.weights ?? modelJSON.weightSpecs ?? []

      return {
        modelTopology: modelJSON.modelTopology ?? null,
        trainingConfig: modelJSON.trainingConfig ?? null,
        weightSpecs: manifestWeights,
        weightData: arrayBuffer
      }
    })

    const model = await tf.loadLayersModel(handler)
    if (!model.optimizer) {
      const optimizer = tf.train.rmsprop(5e-4)
      model.compile({ optimizer, loss: 'categoricalCrossentropy' })
    }
    console.log(`[Brain] Loaded model from ${resolved}`)
    return model
  } catch (err) {
    console.warn('[Brain] No saved model found, creating new one.')
    return null
  }
}

export async function saveBrainState(state, dir = DEFAULT_BRAIN_DIR) {
  const resolved = resolveDir(dir)
  await ensureDir(resolved)
  const statePath = path.join(resolved, 'state.json')
  const payload = {
    ...state,
    savedAt: new Date().toISOString()
  }
  await fs.writeFile(statePath, JSON.stringify(payload, null, 2))
  console.log(`[Brain] Saved state to ${statePath}`)
}

export async function loadBrainState(dir = DEFAULT_BRAIN_DIR) {
  try {
    const resolved = resolveDir(dir)
    const statePath = path.join(resolved, 'state.json')
    const raw = await fs.readFile(statePath, 'utf8')
    return JSON.parse(raw)
  } catch (err) {
    return null
  }
}
