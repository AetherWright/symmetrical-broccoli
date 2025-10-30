import mineflayer from 'mineflayer'
import { Vec3 } from 'vec3'
import {
  createBrain,
  chooseAction,
  trainBrain,
  saveBrain,
  loadBrain,
  saveBrainState,
  loadBrainState,
  DEFAULT_BRAIN_DIR
} from './brain.js'

// ----------------------------
// CONFIG
// ----------------------------
const MC_HOST = 'localhost'
const MC_PORT = 25565
const CRAFTING_ACTIONS = {
  craft_planks: { item: 'oak_planks', amount: 4, allowPartial: true, reward: 0.6 },
  craft_sticks: { item: 'stick', amount: 4, allowPartial: true, reward: 0.45 },
  craft_table: { item: 'crafting_table', amount: 1, allowPartial: false, reward: 0.8 },
  craft_pickaxe: { item: 'stone_pickaxe', amount: 1, requireTable: true, allowPartial: false, reward: 1.2 },
  craft_sword: { item: 'stone_sword', amount: 1, requireTable: true, allowPartial: false, reward: 1.0 },
  craft_shovel: { item: 'stone_shovel', amount: 1, requireTable: true, allowPartial: false, reward: 0.9 },
  craft_torch: { item: 'torch', amount: 4, allowPartial: true, reward: 0.55 },
  craft_furnace: { item: 'furnace', amount: 1, requireTable: true, allowPartial: false, reward: 1.1 }
}
const ACTIONS = [
  'move_forward',
  'move_backward',
  'strafe_left',
  'strafe_right',
  'jump',
  'jump_forward',
  'sprint_forward',
  'sneak_forward',
  'turn_left',
  'turn_right',
  'look_up',
  'look_down',
  'mine',
  'attack',
  'build',
  'build_above',
  'use_item',
  ...Object.keys(CRAFTING_ACTIONS)
]
const TICK_RATE = 1000   // 1 second
const EPSILON_START = 0.25
const EPSILON_MIN = 0.05
const EPSILON_DECAY = 0.999
const SAVE_INTERVAL_TICKS = 40
const SAVE_INTERVAL_MS = 60 * 1000
const CHECKPOINT_DIR = DEFAULT_BRAIN_DIR

// ----------------------------
// INIT BOT + MODEL
// ----------------------------
const bot = mineflayer.createBot({
  host: MC_HOST,
  port: MC_PORT,
  username: 'BrainBot'
})
console.log(`[BrainBot] Connecting to ${MC_HOST}:${MC_PORT}`)

const OBS_SIZE = 22
let brain = null
let brainReady = null
let lastObs = null
let lastAction = null
let lastPos = null
let lastHealth = 20
let lastFood = 20
let lastInvTotal = 0
let epsilon = EPSILON_START
let running = true
let tickTimer = null
let tickInFlight = false
let registry = null
let tickCount = 0
let trainingSteps = 0
let cumulativeReward = 0
let lastSaveTick = 0
let lastSaveTime = Date.now()
let saveInFlight = null
let pendingSaveReason = null

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms))

async function initializeBrain() {
  try {
    const loaded = await loadBrain(CHECKPOINT_DIR)
    if (loaded) {
      brain = loaded
    } else {
      brain = createBrain(OBS_SIZE, ACTIONS.length)
    }

    const savedState = await loadBrainState(CHECKPOINT_DIR)
    if (savedState) {
      if (typeof savedState.epsilon === 'number' && Number.isFinite(savedState.epsilon)) {
        epsilon = Math.min(Math.max(savedState.epsilon, EPSILON_MIN), EPSILON_START)
      }
      if (typeof savedState.tickCount === 'number' && savedState.tickCount >= 0) {
        tickCount = savedState.tickCount
      }
      if (typeof savedState.trainingSteps === 'number' && savedState.trainingSteps >= 0) {
        trainingSteps = savedState.trainingSteps
      }
      if (typeof savedState.cumulativeReward === 'number' && Number.isFinite(savedState.cumulativeReward)) {
        cumulativeReward = savedState.cumulativeReward
      }
      lastSaveTick = tickCount
      if (savedState.savedAt) {
        lastSaveTime = Date.now()
      }
      console.log('[BrainBot] Restored training state from disk.')
    }
  } catch (err) {
    console.error('[BrainBot] Failed to initialize brain from checkpoint:', err)
    brain = createBrain(OBS_SIZE, ACTIONS.length)
  }

  return brain
}

brainReady = initializeBrain()

async function ensureBrainReady() {
  if (brain) return brain
  if (!brainReady) {
    brainReady = initializeBrain()
  }
  try {
    await brainReady
  } catch (err) {
    console.error('[BrainBot] Brain initialization failed, recreating model:', err)
    brain = createBrain(OBS_SIZE, ACTIONS.length)
    brainReady = Promise.resolve(brain)
  }
  return brain
}

async function persistBrain(reason = 'periodic') {
  if (!brain) return
  try {
    await saveBrain(brain, CHECKPOINT_DIR)
    await saveBrainState(
      {
        epsilon,
        tickCount,
        trainingSteps,
        cumulativeReward,
        reason
      },
      CHECKPOINT_DIR
    )
    lastSaveTick = tickCount
    lastSaveTime = Date.now()
    console.log(`[BrainBot] Saved checkpoint (${reason}).`)
  } catch (err) {
    console.error(`[BrainBot] Failed to save checkpoint (${reason}):`, err)
  }
}

function scheduleBrainSave(reason = 'periodic') {
  if (saveInFlight) {
    pendingSaveReason = reason
    return
  }

  saveInFlight = (async () => {
    await persistBrain(reason)
  })()

  saveInFlight
    .catch(err => console.error('[BrainBot] Save task error:', err))
    .finally(() => {
      saveInFlight = null
      if (pendingSaveReason) {
        const nextReason = pendingSaveReason
        pendingSaveReason = null
        scheduleBrainSave(nextReason)
      }
    })
}

function maybeTriggerAutosave() {
  const ticksSinceSave = tickCount - lastSaveTick
  const msSinceSave = Date.now() - lastSaveTime
  if (ticksSinceSave >= SAVE_INTERVAL_TICKS || msSinceSave >= SAVE_INTERVAL_MS) {
    scheduleBrainSave('autosave')
  }
}

async function flushPendingSave() {
  if (saveInFlight) {
    try {
      await saveInFlight
    } catch (err) {
      console.error('[BrainBot] Pending save failed:', err)
    }
  }
}

async function equipBestTool(preferredKeywords = []) {
  const items = bot.inventory?.items?.() ?? []
  for (const keyword of preferredKeywords) {
    const tool = items.find(item => item?.name?.includes(keyword))
    if (tool) {
      try {
        await bot.equip(tool, 'hand')
        return true
      } catch (err) {
        console.warn(`[BrainBot] Failed to equip ${tool.name}:`, err?.message ?? err)
      }
    }
  }
  return false
}

async function equipPlaceableBlock() {
  const items = bot.inventory?.items?.() ?? []
  for (const item of items) {
    if (!item?.name) continue
    if (['sword', 'pickaxe', 'axe', 'shovel', 'hoe', 'bucket'].some(tool => item.name.includes(tool))) continue
    try {
      await bot.equip(item, 'hand')
      return true
    } catch (err) {
      console.warn(`[BrainBot] Failed to equip ${item.name} for building:`, err?.message ?? err)
    }
  }
  return false
}

function findNearbyBlock(name, maxDistance = 4) {
  if (!registry) return null
  const blockInfo = registry.blocksByName?.[name]
  if (!blockInfo) return null
  try {
    return bot.findBlock({ matching: blockInfo.id, maxDistance }) ?? null
  } catch (err) {
    console.warn(`[BrainBot] findNearbyBlock failed for ${name}:`, err?.message ?? err)
    return null
  }
}

async function craftItem(targetName, options = {}) {
  if (!registry) return false

  const {
    amount = 1,
    requireTable = false,
    allowPartial = true,
    tableRange = 4
  } = options

  const itemInfo = registry.itemsByName?.[targetName]
  if (!itemInfo) {
    console.warn(`[BrainBot] Unknown craft target: ${targetName}`)
    return false
  }

  const tableBlock = findNearbyBlock('crafting_table', tableRange)
  const candidates = []

  if (tableBlock) {
    candidates.push(tableBlock)
  }
  if (!requireTable || !tableBlock) {
    candidates.push(null)
  }

  if (requireTable && !tableBlock) {
    console.log('[BrainBot] Crafting table required but not nearby.')
    return false
  }

  const uniqueCandidates = []
  for (const candidate of candidates) {
    if (!uniqueCandidates.some(existing => existing === candidate)) {
      uniqueCandidates.push(candidate)
    }
  }

  for (const table of uniqueCandidates) {
    let craftCount = Math.max(1, Math.floor(amount))
    const minCount = allowPartial ? 1 : craftCount

    while (craftCount >= minCount) {
      try {
        const recipes = bot.recipesFor(itemInfo.id, null, craftCount, table ?? null)
        if (recipes?.length) {
          try {
            await bot.craft(recipes[0], craftCount, table ?? undefined)
            return true
          } catch (err) {
            console.warn(`[BrainBot] Craft ${targetName} x${craftCount} failed:`, err?.message ?? err)
          }
        }
      } catch (err) {
        console.warn(`[BrainBot] recipesFor failed for ${targetName}:`, err?.message ?? err)
      }

      if (!allowPartial) break
      if (craftCount === 1) break
      craftCount = Math.max(1, Math.floor(craftCount / 2))
      if (craftCount === minCount && !allowPartial) break
      if (craftCount === 1 && !allowPartial) break
    }
  }

  return false
}

async function executeCraftAction(act) {
  const config = CRAFTING_ACTIONS[act]
  if (!config) return

  const success = await craftItem(config.item, config)
  if (success) {
    blockReward += config.reward ?? 0.4
    console.log(`[BrainBot] Crafted ${config.item}`)
  } else {
    blockReward -= 0.03
  }
}

// ----------------------------
// OBSERVATION GATHERING
// ----------------------------
function gatherObservations(bot) {
  const obs = new Float32Array(OBS_SIZE)

  if (!bot?.entity?.position) {
    return obs
  }

  const pos = bot.entity.position
  const vel = bot.entity.velocity ?? { x: 0, y: 0, z: 0 }
  const yaw = bot.entity?.yaw ?? 0
  const pitch = bot.entity?.pitch ?? 0
  const entities = Object.values(bot.entities ?? {})
  const inv = (bot.inventory?.slots ?? []).filter(Boolean)

  let nearestEntityDist = 0
  if (entities.length) {
    const selfPos = bot.entity.position
    let closest = Infinity
    for (const ent of entities) {
      if (!ent?.position || ent === bot.entity) continue
      const dx = ent.position.x - selfPos.x
      const dy = ent.position.y - selfPos.y
      const dz = ent.position.z - selfPos.z
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz)
      if (dist < closest) {
        closest = dist
      }
    }
    nearestEntityDist = Number.isFinite(closest) ? closest : 0
  }

  const invTotal = inv.reduce((sum, item) => sum + (item?.count ?? 0), 0)

  obs[0] = Number(pos.x) || 0
  obs[1] = Number(pos.y) || 0
  obs[2] = Number(pos.z) || 0
  obs[3] = Number(vel.x) || 0
  obs[4] = Number(vel.y) || 0
  obs[5] = Number(vel.z) || 0
  obs[6] = Number(yaw) || 0
  obs[7] = Number(pitch) || 0
  obs[8] = Number(bot.health ?? 20) || 0
  obs[9] = Number(bot.food ?? 20) || 0
  obs[10] = Number(bot.oxygenLevel ?? bot.oxygen ?? 20) || 0
  obs[11] = bot.entity?.onGround ? 1 : 0
  obs[12] = bot.controlState?.sprint ? 1 : 0
  obs[13] = bot.controlState?.sneak ? 1 : 0
  obs[14] = Number(bot.time?.age ?? 0) || 0
  obs[15] = entities.length
  obs[16] = nearestEntityDist
  obs[17] = inv.length
  obs[18] = invTotal
  obs[19] = Number(bot.quickBarSlot ?? 0)
  obs[20] = Number(bot.experience?.level ?? 0)
  obs[21] = Number(bot.experience?.progress ?? 0)

  return obs
}

// ----------------------------
// ACTION EXECUTION
// ----------------------------
async function executeAction(index) {
  const act = ACTIONS[index]
  if (!act) return
  console.log(`[BrainBot] Executing: ${act}`)

  if (CRAFTING_ACTIONS[act]) {
    await executeCraftAction(act)
    bot.clearControlStates()
    return
  }

  const holdControls = async (states = [], duration = 350) => {
    for (const state of states) {
      bot.setControlState(state, true)
    }
    await sleep(duration)
    for (const state of states) {
      bot.setControlState(state, false)
    }
  }

  const lookBy = async (deltaYaw = 0, deltaPitch = 0) => {
    const yaw = bot.entity?.yaw ?? 0
    const pitch = bot.entity?.pitch ?? 0
    const nextYaw = yaw + deltaYaw
    const nextPitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, pitch + deltaPitch))
    await bot.look(nextYaw, nextPitch, true)
  }

  const getTargetBlock = () => bot.blockAtCursor(5)

  try {
    switch (act) {
      case 'move_forward':
        await holdControls(['forward'])
        break
      case 'move_backward':
        await holdControls(['back'])
        break
      case 'strafe_left':
        await holdControls(['left'])
        break
      case 'strafe_right':
        await holdControls(['right'])
        break
      case 'jump':
        await holdControls(['jump'])
        break
      case 'jump_forward':
        await holdControls(['jump', 'forward'])
        break
      case 'sprint_forward':
        await holdControls(['forward', 'sprint'], 450)
        break
      case 'sneak_forward':
        await holdControls(['forward', 'sneak'], 450)
        break
      case 'turn_left':
        await lookBy(-Math.PI / 6, 0)
        break
      case 'turn_right':
        await lookBy(Math.PI / 6, 0)
        break
      case 'look_up':
        await lookBy(0, -Math.PI / 12)
        break
      case 'look_down':
        await lookBy(0, Math.PI / 12)
        break
      case 'mine': {
        const block = getTargetBlock()
        if (block) {
          await equipBestTool(['pickaxe', 'axe', 'shovel'])
          try {
            await bot.dig(block)
          } catch (err) {
            console.warn('[BrainBot] Dig action failed:', err?.message ?? err)
          }
        }
        break
      }
      case 'attack': {
        const target = bot.nearestEntity()
        if (target) {
          await equipBestTool(['sword', 'axe'])
          try {
            bot.attack(target)
            blockReward += 0.1
          } catch (err) {
            console.warn('[BrainBot] Attack action failed:', err?.message ?? err)
          }
        }
        break
      }
      case 'build': {
        const ref = getTargetBlock()
        if (ref) {
          await equipPlaceableBlock()
          try {
            await bot.placeBlock(ref, new Vec3(0, 1, 0))
            blockReward += 0.4
          } catch (err) {
            console.warn('[BrainBot] Build action failed:', err?.message ?? err)
          }
        }
        break
      }
      case 'build_above': {
        const basePos = bot.entity?.position?.floored?.() ?? bot.entity?.position
        const belowPos = basePos ? basePos.offset(0, -1, 0) : null
        const below = belowPos ? bot.blockAt(belowPos) : null
        if (below) {
          await equipPlaceableBlock()
          try {
            await bot.placeBlock(below, new Vec3(0, 1, 0))
            blockReward += 0.4
          } catch (err) {
            console.warn('[BrainBot] Build-above action failed:', err?.message ?? err)
          }
        }
        break
      }
      case 'use_item': {
        try {
          bot.activateItem()
          await sleep(300)
          bot.deactivateItem()
          blockReward += 0.05
        } catch (err) {
          console.warn('[BrainBot] Use-item action failed:', err?.message ?? err)
        }
        break
      }
    }
  } finally {
    bot.clearControlStates()
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

bot.on('playerCollect', (collector, collected) => {
  if (collector === bot.entity) {
    const count = collected?.metadata?.itemCount ?? collected?.count ?? 1
    const bonus = Math.max(0.3, (count || 1) * 0.15)
    blockReward += bonus
    console.log(`[BrainBot] Collected item → +${bonus.toFixed(2)} reward`)
  }
})

// revised computeReward
function computeReward(obs) {
  let reward = 0

  const pos = { x: obs[0], y: obs[1], z: obs[2] }
  if (lastPos) {
    const dx = pos.x - lastPos.x
    const dy = pos.y - lastPos.y
    const dz = pos.z - lastPos.z
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz)
    const horizontal = Math.sqrt(dx * dx + dz * dz)
    reward += Math.min(dist * 0.1, 0.5)
    reward += Math.min(horizontal * 0.05, 0.25)
    reward += Math.min(Math.abs(dy) * 0.05, 0.15)
  }

  const health = obs[8]
  if (Number.isFinite(health)) {
    if (health < lastHealth) {
      reward -= Math.min(1, (lastHealth - health) * 0.5)
    } else if (health > lastHealth) {
      reward += Math.min(1, (health - lastHealth) * 0.5)
    }
    lastHealth = health
  }

  const food = obs[9]
  if (Number.isFinite(food)) {
    if (food > lastFood) {
      reward += Math.min(0.5, (food - lastFood) * 0.1)
    } else if (food < lastFood) {
      reward -= Math.min(0.5, (lastFood - food) * 0.05)
    }
    lastFood = food
  }

  const invTotal = obs[18]
  if (Number.isFinite(invTotal)) {
    const delta = invTotal - lastInvTotal
    if (delta !== 0) {
      reward += Math.sign(delta) * Math.min(Math.abs(delta) * 0.2, 1.5)
    }
    lastInvTotal = invTotal
  }

  const nearestDist = obs[16]
  if (Number.isFinite(nearestDist) && nearestDist > 0 && nearestDist < 3) {
    reward -= (3 - nearestDist) * 0.05
  }

  const consumedBlockReward = blockReward
  reward += consumedBlockReward
  blockReward = 0

  reward -= 0.02 // mild time penalty to encourage efficiency

  lastPos = { ...pos }
  return reward
}


// ----------------------------
// CUSTOM TICK LOOP
// ----------------------------
async function tickLoop() {
  if (!running || tickInFlight) return
  tickInFlight = true

  try {
    await ensureBrainReady()
    if (!bot?.entity?.position) {
      console.warn('[BrainBot] Entity not ready, skipping tick.')
      return
    }

    if (!registry && bot.registry) {
      registry = bot.registry
    }

    const observation = gatherObservations(bot)
    const reward = computeReward(observation)

    let trained = false
    if (lastObs && lastAction != null) {
      trained = await trainBrain(brain, lastObs, lastAction, reward)
    }

    const action = await chooseAction(brain, observation, epsilon)
    await executeAction(action)

    lastObs = observation
    lastAction = action

    tickCount += 1
    cumulativeReward += reward
    if (trained) {
      trainingSteps += 1
    }

    if (epsilon > EPSILON_MIN) {
      epsilon = Math.max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    }

    console.log(`[BrainBot] Tick done | Reward: ${reward.toFixed(3)} | Eps: ${epsilon.toFixed(3)}`)
    maybeTriggerAutosave()
  } catch (err) {
    console.error('[BrainBot] Tick error:', err)
  } finally {
    tickInFlight = false

    if (running) {
      tickTimer = setTimeout(() => {
        tickTimer = null
        tickLoop().catch(err => console.error('[BrainBot] Tick scheduling error:', err))
      }, TICK_RATE)
    }
  }
}

bot.once('spawn', () => {
  console.log('[BrainBot] Spawned! Waiting for entity to initialize...')

  registry = bot.registry ?? registry
  if (!registry) {
    console.warn('[BrainBot] Failed to load registry — crafting actions will be limited.')
  }

  const waitForEntity = setInterval(() => {
    if (bot?.entity?.position) {
      clearInterval(waitForEntity)
      console.log('[BrainBot] Entity ready — starting tick loop!')

      // start the main loop
      tickLoop().catch(err => console.error('[BrainBot] Initial tick error:', err))
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
  console.log('[BrainBot] Type "pause", "resume", "save", or "exit".')

  process.stdin.on('data', async data => {
    const cmd = data.trim().toLowerCase()
    if (cmd === 'pause') {
      running = false
    if (tickTimer) {
      clearTimeout(tickTimer)
      tickTimer = null
    }
    console.log('[BrainBot] Paused. Current tick will finish before stopping.')
  } else if (cmd === 'resume') {
    if (!running) {
      running = true
      console.log('[BrainBot] Resumed.')
      if (!tickInFlight && !tickTimer) {
        tickLoop().catch(err => console.error('[BrainBot] Resume tick error:', err))
      }
    }
  } else if (cmd === 'save') {
    console.log('[BrainBot] Manual save requested...')
    scheduleBrainSave('manual')
  } else if (['exit','quit','stop'].includes(cmd)) {
    console.log('[BrainBot] Saving model + shutting down...')
    await ensureBrainReady()
    await flushPendingSave()
    await persistBrain('shutdown')
    running = false
    if (tickTimer) {
      clearTimeout(tickTimer)
      tickTimer = null
    }
    bot.quit('Manual shutdown')
    process.exit(0)
  }
})

async function gracefulShutdown(reason = 'signal') {
  try {
    console.log(`[BrainBot] Caught ${reason}. Saving before exit...`)
    running = false
    if (tickTimer) {
      clearTimeout(tickTimer)
      tickTimer = null
    }
    await ensureBrainReady()
    await flushPendingSave()
    await persistBrain(reason)
    if (bot) {
      try {
        bot.quit(`Shutdown: ${reason}`)
      } catch (err) {
        console.warn('[BrainBot] Failed to quit bot during shutdown:', err)
      }
    }
  } catch (err) {
    console.error('[BrainBot] Failed during graceful shutdown:', err)
  } finally {
    process.exit(0)
  }
}

process.on('SIGINT', () => gracefulShutdown('SIGINT'))
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'))
