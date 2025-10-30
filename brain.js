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

function ensureBrainOptimizers(model) {
  if (!model) return

  if (!model.actionOptimizer) {
    model.actionOptimizer = tf.train.rmsprop(5e-4)
  }

  if (!model.predictionOptimizer) {
    model.predictionOptimizer = tf.train.adam(1e-3)
  }

  if (!Number.isFinite(model.hebbianRate)) {
    model.hebbianRate = 5e-4
  }

  if (!model.featureExtractor) {
    attachFeatureExtractor(model)
  }

  if (!model.optimizer) {
    model.compile({
      optimizer: model.actionOptimizer,
      loss: { action_head: 'categoricalCrossentropy', prediction_head: 'meanSquaredError' },
      lossWeights: { action_head: 1, prediction_head: 0 }
    })
  }
}

function attachFeatureExtractor(model) {
  if (!model) return
  try {
    const policyLayer = model.getLayer('policy_features')
    if (policyLayer && model.inputs?.length) {
      model.featureExtractor = tf.model({ inputs: model.inputs[0], outputs: policyLayer.output })
    }
  } catch (err) {
    model.featureExtractor = null
  }
}

// ----------------------------
// MODEL CREATION
// ----------------------------
export function createBrain(inputSize, actionCount) {
  const input = tf.input({ shape: [inputSize] })

  let features = input

  if (BRAIN_CONFIG.useLSTM) {
    const reshaped = tf.layers.reshape({ targetShape: [1, inputSize] }).apply(features)
    const lstm = tf.layers
      .lstm({
        units: BRAIN_CONFIG.hiddenUnits,
        activation: 'tanh',
        recurrentActivation: 'sigmoid',
        returnSequences: false
      })
      .apply(reshaped)
    features = tf.layers.batchNormalization().apply(lstm)
  } else {
    const normalized = tf.layers.batchNormalization().apply(features)
    features = tf.layers
      .dense({
        units: BRAIN_CONFIG.hiddenUnits,
        activation: 'relu'
      })
      .apply(normalized)
  }

  const dropped = tf.layers.dropout({ rate: BRAIN_CONFIG.dropoutRate }).apply(features)
  const sharedDense = tf.layers
    .dense({
      units: Math.max(1, Math.floor(BRAIN_CONFIG.hiddenUnits / 2)),
      activation: 'relu',
      name: 'shared_dense'
    })
    .apply(dropped)
  const policyFeatures = tf.layers
    .dropout({ rate: BRAIN_CONFIG.dropoutRate, name: 'policy_features' })
    .apply(sharedDense)

  const actionHead = tf.layers
    .dense({ units: actionCount, activation: 'softmax', name: 'action_head' })
    .apply(policyFeatures)
  const predictionHead = tf.layers
    .dense({ units: inputSize, activation: 'linear', name: 'prediction_head' })
    .apply(policyFeatures)

  const model = tf.model({ inputs: input, outputs: [actionHead, predictionHead] })
  attachFeatureExtractor(model)
  ensureBrainOptimizers(model)

  console.log(`[Brain] Created model with ${BRAIN_CONFIG.hiddenUnits} hidden units`)
  return model
}

// ----------------------------
// ACTION SELECTION
// ----------------------------
export async function chooseAction(model, obs, epsilon = 0.1) {
  const probsArray = tf.tidy(() => {
    const input = tf.tensor(obs, [1, obs.length], 'float32')
    const outputs = model.predict(input)
    let actionTensor = outputs
    if (Array.isArray(outputs)) {
      actionTensor = outputs[0]
      for (let i = 1; i < outputs.length; i++) {
        outputs[i].dispose?.()
      }
    }
    const data = actionTensor.dataSync()
    actionTensor.dispose?.()
    return data
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
export async function trainBrain(model, obs, actionIndex, reward, nextObservation) {
  ensureBrainOptimizers(model)

  const actionOutput = Array.isArray(model.outputs) ? model.outputs[0] : model.outputs
  const rawUnits = actionOutput?.shape ? actionOutput.shape[actionOutput.shape.length - 1] : null

  if (!Number.isFinite(rawUnits) || rawUnits <= 0) {
    return false
  }

  const outputUnits = Math.max(1, Math.floor(rawUnits))
  const boundedActionIndex = Math.max(0, Math.min(outputUnits - 1, Math.floor(actionIndex)))
  const scaledReward = Math.max(-1, Math.min(1, reward ?? 0))
  const shouldTrainPolicy = Number.isFinite(scaledReward) && scaledReward !== 0
  const hasNextObservation =
    Array.isArray(nextObservation) && nextObservation.length === obs.length

  if (!shouldTrainPolicy && !hasNextObservation) {
    return false
  }

  if (typeof tf.nextFrame === 'function') {
    await tf.nextFrame()
  }

  let trained = false

  tf.tidy(() => {
    const xs = tf.tensor(obs, [1, obs.length], 'float32')
    const labelBuffer = new Float32Array(outputUnits)
    if (Number.isFinite(boundedActionIndex)) {
      labelBuffer[boundedActionIndex] = 1
    }
    const ys = tf.tensor(labelBuffer, [1, outputUnits], 'float32')
    const nextTensor = hasNextObservation
      ? tf.tensor(nextObservation, [1, nextObservation.length], 'float32')
      : null

    if (shouldTrainPolicy) {
      const lossValue = model.actionOptimizer.minimize(() => {
        const outputs = model.apply(xs, { training: true })
        const actionTensor = Array.isArray(outputs) ? outputs[0] : outputs
        const logProbs = actionTensor.log()
        const selectedLogProb = logProbs.mul(ys).sum(-1)
        const scaledLoss = selectedLogProb.mul(-scaledReward)
        if (Array.isArray(outputs)) {
          for (let i = 1; i < outputs.length; i++) {
            outputs[i].dispose?.()
          }
        }
        return scaledLoss.mean()
      }, true)
      if (lossValue) {
        lossValue.dispose()
      }
      trained = true
    }

    if (nextTensor) {
      const predictionLoss = model.predictionOptimizer.minimize(() => {
        const outputs = model.apply(xs, { training: true })
        const predictionTensor = Array.isArray(outputs) ? outputs[1] : outputs
        if (!predictionTensor) {
          return tf.scalar(0)
        }
        const diff = predictionTensor.sub(nextTensor)
        const mse = diff.square().mean()
        diff.dispose()
        if (Array.isArray(outputs)) {
          outputs[0]?.dispose?.()
          for (let i = 2; i < outputs.length; i++) {
            outputs[i].dispose?.()
          }
        }
        return mse
      }, true)
      if (predictionLoss) {
        predictionLoss.dispose()
      }
      trained = true
    }

    if (model.hebbianRate > 0) {
      const outputs = model.predict(xs)
      const actionTensor = Array.isArray(outputs) ? outputs[0] : outputs
      const predictionTensor = Array.isArray(outputs) ? outputs[1] : null
      applyHebbianUpdate(model, xs, actionTensor, nextTensor, predictionTensor)
      if (Array.isArray(outputs)) {
        for (let i = 0; i < outputs.length; i++) {
          outputs[i].dispose?.()
        }
      } else {
        outputs.dispose?.()
      }
    }

    ys.dispose()
    nextTensor?.dispose()
  })

  return trained
}

function applyHebbianUpdate(model, inputTensor, actionTensor, nextTensor, predictionTensor) {
  const rate = Number(model.hebbianRate)
  if (!Number.isFinite(rate) || rate <= 0) {
    return
  }

  const extractor = model.featureExtractor
  if (!extractor) {
    return
  }

  try {
    const features = extractor.predict(inputTensor)
    if (!features) {
      return
    }

    const featureSize = features.shape?.[features.shape.length - 1]
    if (!Number.isFinite(featureSize) || featureSize <= 0) {
      features.dispose?.()
      return
    }

    const featureVector = features.reshape([featureSize])
    const featureColumn = featureVector.expandDims(1)

    const actionUnits = actionTensor?.shape?.[actionTensor.shape.length - 1]
    if (!Number.isFinite(actionUnits) || actionUnits <= 0) {
      featureColumn.dispose()
      featureVector.dispose()
      features.dispose()
      return
    }

    const actionVector = actionTensor.reshape([actionUnits])
    const actionRow = actionVector.expandDims(0)
    const hebbianOuter = featureColumn.matMul(actionRow).mul(rate)

    const actionLayer = safeGetLayer(model, 'action_head')
    if (actionLayer) {
      const [kernel, bias] = actionLayer.getWeights()
      const updatedKernel = kernel.add(hebbianOuter)
      const updatedBias = bias.clone()
      actionLayer.setWeights([updatedKernel, updatedBias])
      kernel.dispose()
      bias.dispose()
    }

    hebbianOuter.dispose()
    actionRow.dispose()
    actionVector.dispose()

    if (nextTensor && predictionTensor) {
      const targetUnits = nextTensor.shape?.[nextTensor.shape.length - 1]
      const predictionUnits = predictionTensor.shape?.[predictionTensor.shape.length - 1]

      if (Number.isFinite(targetUnits) && targetUnits > 0 && targetUnits === predictionUnits) {
        const predictionLayer = safeGetLayer(model, 'prediction_head')
        if (predictionLayer) {
          const targetVector = nextTensor.reshape([targetUnits])
          const predictionVector = predictionTensor.reshape([predictionUnits])
          const errorVector = targetVector.sub(predictionVector)
          const predictionRow = errorVector.expandDims(0)
          const predictionOuter = featureColumn.matMul(predictionRow).mul(rate * 0.5)

          const [predKernel, predBias] = predictionLayer.getWeights()
          const updatedKernel = predKernel.add(predictionOuter)
          const biasAdjustment = errorVector.mul(rate * 0.1)
          const updatedBias = predBias.add(biasAdjustment)
          predictionLayer.setWeights([updatedKernel, updatedBias])
          predKernel.dispose()
          predBias.dispose()
          predictionOuter.dispose()
          predictionRow.dispose()
          biasAdjustment.dispose()
          errorVector.dispose()
          targetVector.dispose()
          predictionVector.dispose()
        }
      }
    }

    featureColumn.dispose()
    featureVector.dispose()
    features.dispose()
  } catch (err) {
    console.warn('[Brain] Hebbian update skipped due to error:', err)
  }
}

function safeGetLayer(model, name) {
  try {
    return model?.getLayer?.(name)
  } catch (err) {
    return null
  }
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

    const loadedModel = await tf.loadLayersModel(handler)
    let finalModel = loadedModel

    if (!Array.isArray(loadedModel.outputs) || loadedModel.outputs.length < 2) {
      console.warn('[Brain] Loaded legacy single-head model, migrating to dual-head architecture.')
      const inputShape = loadedModel.inputs?.[0]?.shape
      const actionShape = loadedModel.outputs?.[0]?.shape
      const inputSize = inputShape ? inputShape[inputShape.length - 1] : null
      const actionUnits = actionShape ? actionShape[actionShape.length - 1] : null

      if (Number.isFinite(inputSize) && Number.isFinite(actionUnits)) {
        const upgraded = createBrain(inputSize, actionUnits)
        const legacyWeights = loadedModel.getWeights()
        const upgradedWeights = upgraded.getWeights()
        const assignable = Math.min(legacyWeights.length, upgradedWeights.length)
        const weightsToAssign = upgradedWeights.map((tensor, idx) => {
          if (idx < assignable) {
            tensor.dispose()
            return legacyWeights[idx]
          }
          return tensor
        })
        upgraded.setWeights(weightsToAssign)
        for (let i = assignable; i < legacyWeights.length; i++) {
          legacyWeights[i].dispose()
        }
        finalModel = upgraded
        loadedModel.dispose()
      } else {
        console.warn('[Brain] Could not determine legacy model dimensions. Creating fresh brain.')
        loadedModel.dispose()
        return null
      }
    }

    ensureBrainOptimizers(finalModel)
    console.log(`[Brain] Loaded model from ${resolved}`)
    return finalModel
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
