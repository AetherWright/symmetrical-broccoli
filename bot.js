import mineflayer from 'mineflayer'
import { Vec3 } from 'vec3'
import * as tf from '@tensorflow/tfjs'
import { createBrain, chooseAction, trainBrain } from './brain.js'

// ----------------------------
// CONFIG
// ----------------------------
const MC_HOST = 'localhost'
const MC_PORT = 25565
const ACTIONS = ['move_forward', 'turn_left', 'turn_right', 'mine', 'attack', 'build']
const TICK_RATE = 1000   // 1 second
const EPSILON_START = 0.25

// ----------------------------
// INIT BOT + MODEL
// ----------------------------
const bot = mineflayer.createBot({
  host: MC_HOST,
  port: MC_PORT,
  username: 'BrainBot'
})
console.log(`[BrainBot] Connecting to ${MC_HOST}:${MC_PORT}`)

const OBS_SIZE = 12
const brain = createBrain(OBS_SIZE, ACTIONS.length)
let lastObs = null
let lastAction = null
let lastPos = null
let lastHealth = 20
let epsilon = EPSILON_START
let running = true

// ----------------------------
// OBSERVATION GATHERING
// ----------------------------
function gatherObservations(bot) {
  if (!bot?.entity?.position) {
    return Array(800).fill(0)
  }

  const pos = bot.entity.position
  const vel = bot.entity.velocity ?? { x: 0, y: 0, z: 0 }
  const entities = Object.values(bot.entities)
  const inv = bot.inventory.slots.filter(Boolean)

  return [
    pos.x, pos.y, pos.z,
    vel.x, vel.y, vel.z,
    bot.health ?? 20,
    bot.food ?? 20,
    bot.time?.age ?? 0,
    entities.length,
    inv.length,
    inv.reduce((sum, i) => sum + (i.count || 0), 0)
  ]
}

// ----------------------------
// ACTION EXECUTION
// ----------------------------
async function executeAction(index) {
  const act = ACTIONS[index]
  if (!act) return
  console.log(`[BrainBot] Executing: ${act}`)

  switch (act) {
    case 'move_forward':
      bot.setControlState('forward', true)
      setTimeout(() => bot.clearControlStates(), 300)
      break
    case 'turn_left':
      bot.entity.yaw -= Math.PI / 8
      break
    case 'turn_right':
      bot.entity.yaw += Math.PI / 8
      break
    case 'mine': {
      const block = bot.blockAtCursor()
      if (block) await bot.dig(block)
      break
    }
    case 'attack': {
      const target = bot.nearestEntity()
      if (target) bot.attack(target)
      break
    }
    case 'build': {
      const ref = bot.blockAtCursor()
      if (ref) await bot.placeBlock(ref, new Vec3(0, 1, 0))
      break
    }
  }
}

// ----------------------------
// REWARD FUNCTION
// ----------------------------
let blockReward = 0

// only reward real block breaks
bot.on('blockBreak', (block) => {
  if (!block || block.name === 'air') return
  // give more reward for "real" blocks
  const value =
    block.name.includes('ore') ? 2.0 :
    block.name.includes('stone') ? 1.0 :
    block.name.includes('dirt') ? 0.5 :
    0.3
  blockReward += value
  console.log(`[BrainBot] Broke ${block.name} → +${value.toFixed(2)} reward`)
})

// light penalty for failed dig attempts
bot.on('diggingAborted', () => {
  blockReward -= 0.1
  console.log('[BrainBot] Dig aborted → -0.1 penalty')
})

// revised computeReward
function computeReward(obs) {
  let reward = 0

  // movement reward
  const pos = { x: obs[0], y: obs[1], z: obs[2] }
  if (lastPos) {
    const dx = pos.x - lastPos.x
    const dy = pos.y - lastPos.y
    const dz = pos.z - lastPos.z
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz)
    reward += Math.min(dist * 0.1, 0.5)
  }

  // health penalty
  const health = obs[6]
  if (health < lastHealth) reward -= 1
  lastHealth = health

  // add in any mining rewards accumulated since last tick
  reward += blockReward
  blockReward = 0 // reset after consuming

  // small exploration noise
  reward += Math.random() * 0.05

  lastPos = { ...pos }
  return reward
}


// ----------------------------
// CUSTOM TICK LOOP
// ----------------------------
async function tickLoop(bot, obs) {
  if (!running) return
  try {
    if (!bot?.entity?.position) {
      console.warn('[BrainBot] Entity not ready, skipping tick.')
      return
    }

    const newObs = gatherObservations(bot)
    const action = await chooseAction(brain, newObs, epsilon)
    await executeAction(action)

    const reward = computeReward(newObs)

    if (lastObs && lastAction != null) {
      await trainBrain(brain, lastObs, lastAction, 1e-3)
    }

    lastObs = newObs
    lastAction = action

    if (epsilon > 0.05) epsilon *= 0.999  // decay exploration
    console.log(`[BrainBot] Tick done | Reward: ${reward.toFixed(3)} | Eps: ${epsilon.toFixed(3)}`)

  } catch (err) {
    console.error('[BrainBot] Tick error:', err)
  }

  // recursive scheduling
  setTimeout(() => tickLoop(bot), TICK_RATE)
}

bot.once('spawn', () => {
  console.log('[BrainBot] Spawned! Waiting for entity to initialize...')

  const waitForEntity = setInterval(() => {
    if (bot?.entity?.position) {
      clearInterval(waitForEntity)
      console.log('[BrainBot] Entity ready — starting tick loop!')

      // start the main loop
      tickLoop(bot)
    }
  }, 500)
})


// ----------------------------
// ERROR + SHUTDOWN
// ----------------------------
bot.on('error', e => console.error('[BrainBot] Error:', e))
bot.on('kicked', r => console.error('[BrainBot] Kicked:', r))

process.stdin.resume()
process.stdin.setEncoding('utf8')
console.log('[BrainBot] Type "pause", "resume", or "exit".')

process.stdin.on('data', async data => {
  const cmd = data.trim().toLowerCase()
  if (cmd === 'pause') {
    running = false
    console.log('[BrainBot] Paused.')
  } else if (cmd === 'resume') {
    if (!running) {
      running = true
      console.log('[BrainBot] Resumed.')
      tickLoop()
    }
  } else if (['exit','quit','stop'].includes(cmd)) {
    console.log('[BrainBot] Saving model + shutting down...')
    await brain.save('file://./tf_brain_checkpoint')
    running = false
    bot.quit('Manual shutdown')
    process.exit(0)
  }
})
