// brain.js
import * as tf from '@tensorflow/tfjs'

// ----------------------------
// CONFIG
// ----------------------------
export const BRAIN_CONFIG = {
  hiddenUnits: 64,
  dropoutRate: 0.2,
  useLSTM: true,   // flip this if you want temporal memory
  clipNorm: 1.0
}

// ----------------------------
// MODEL CREATION
// ----------------------------
export function createBrain(inputSize, actionCount) {
  const model = tf.sequential()

  // Normalize input
  model.add(tf.layers.batchNormalization({ inputShape: [inputSize] }))

  // Main hidden stack
  if (BRAIN_CONFIG.useLSTM) {
    // Wrap the input for time steps = 1
    model.add(tf.layers.reshape({ targetShape: [1, inputSize] }))
    model.add(tf.layers.lstm({
      units: BRAIN_CONFIG.hiddenUnits,
      activation: 'tanh',
      recurrentActivation: 'sigmoid',
      returnSequences: false
    }))
  } else {
    model.add(tf.layers.dense({
      units: BRAIN_CONFIG.hiddenUnits,
      activation: 'relu'
    }))
  }

  // Optional deeper stack
  model.add(tf.layers.dropout({ rate: BRAIN_CONFIG.dropoutRate }))
  model.add(tf.layers.dense({ units: BRAIN_CONFIG.hiddenUnits / 2, activation: 'relu' }))
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
  const input = tf.tensor([obs])
  const probs = (await model.predict(input).array())[0]
  input.dispose()

  // epsilon-greedy exploration
  if (Math.random() < epsilon) {
    return Math.floor(Math.random() * probs.length)
  }
  return probs.indexOf(Math.max(...probs))
}

// ----------------------------
// TRAINING
// ----------------------------
export async function trainBrain(model, obs, actionIndex, reward, lr = 1e-3) {
  const xs = tf.tensor([obs])
  const ys = tf.oneHot(tf.tensor1d([actionIndex], 'int32'), model.outputs[0].shape[1])

  // scaled reward adjustment (RL-style)
  const scaledReward = Math.max(-1, Math.min(1, reward || 0))

  const {grads} = tf.variableGrads(() => {
    const pred = model.predict(xs)
    const loss = tf.losses.softmaxCrossEntropy(ys, pred).mul(-scaledReward)
    return loss
  })
  model.optimizer.applyGradients(grads)

}

// ----------------------------
// PERSISTENCE
// ----------------------------
export async function saveBrain(model, path = 'file://./brain_checkpoint') {
  await model.save(path)
  console.log(`[Brain] Saved model to ${path}`)
}

export async function loadBrain(path = 'file://./brain_checkpoint') {
  try {
    const model = await tf.loadLayersModel(path + '/model.json')
    console.log(`[Brain] Loaded model from ${path}`)
    return model
  } catch (err) {
    console.warn('[Brain] No saved model found, creating new one.')
    return null
  }
}
