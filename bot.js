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
} from './brainClient.js'

// ----------------------------
// CONFIG
// ----------------------------
const MC_HOST = 'localhost'
const MC_PORT = 25565
const BOT_COUNT = Math.max(2, parseInt(process.env.BOT_COUNT ?? '3', 10))
const GENERATION_TICKS = Math.max(50, parseInt(process.env.GENERATION_TICKS ?? '200', 10))
const ROCK_PARTS = ['Rock', 'Stone', 'Grav', 'Ore', 'Pebble', 'Granite', 'Basalt', 'Iron', 'Coal', 'Quartz']
const SUFFIXES = ['son', 'grip', 'deep', 'delver', 'breaker', 'forge', 'drill', 'hammer', 'core', 'blast']
const PREFIXES = ['', 'Mc', 'Von', 'De', "O'", 'El']

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

const TICK_RATE = 1000
const EPSILON_START = 0.25
const EPSILON_MIN = 0.05
const EPSILON_DECAY = 0.999
const SAVE_INTERVAL_TICKS = 40
const SAVE_INTERVAL_MS = 60 * 1000
const CHECKPOINT_DIR = DEFAULT_BRAIN_DIR
const OBS_SIZE = 22
const MAX_USERNAME_LENGTH = 16

// ----------------------------
// GLOBAL STATE
// ----------------------------
const contexts = []
const usedNames = new Set()
let globalRunning = true
let baselineBrain = null
let baselineReady = null
let baselineState = {
  epsilon: EPSILON_START,
  tickCount: 0,
  trainingSteps: 0,
  cumulativeReward: 0,
  generation: 0
}
let lastSaveTick = 0
let lastSaveTime = Date.now()
let saveInFlight = null
let pendingSaveReason = null
let generationSyncInFlight = false

// ----------------------------
// HELPERS
// ----------------------------
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms))

function sanitizeNetworkString(value, {
  fallback = '',
  maxLength = 64,
  allowed = /[0-9A-Za-z_\-]/,
  label = 'value'
} = {}) {
  let raw = ''
  if (typeof value === 'string') {
    raw = value
  } else if (value != null && typeof value.toString === 'function') {
    raw = value.toString()
  }

  if (!raw) {
    if (!fallback) return ''
    const trimmedFallback = fallback.slice(0, maxLength)
    console.warn(`[NetworkGuard] Missing ${label}, using fallback "${trimmedFallback}".`)
    return trimmedFallback
  }

  let sanitized = ''
  const normalized = raw.normalize('NFKC')
  for (const ch of normalized) {
    if (allowed instanceof RegExp) {
      allowed.lastIndex = 0
      if (!allowed.test(ch)) continue
    }
    sanitized += ch
    if (sanitized.length >= maxLength) break
  }

  if (!sanitized) {
    if (!fallback) {
      console.warn(`[NetworkGuard] Dropped invalid ${label} "${raw}".`)
      return ''
    }
    const trimmedFallback = fallback.slice(0, maxLength)
    console.warn(`[NetworkGuard] Sanitized ${label} "${raw}" to fallback "${trimmedFallback}".`)
    return trimmedFallback
  }

  if (sanitized !== raw) {
    console.warn(`[NetworkGuard] Sanitized ${label}: "${raw}" → "${sanitized}".`)
  }

  return sanitized
}

function sanitizeNetworkPort(value, fallback = 25565) {
  const numeric = Number(value)
  if (Number.isInteger(numeric) && numeric >= 1 && numeric <= 65535) {
    return numeric
  }
  console.warn(`[NetworkGuard] Invalid port "${value}". Using fallback ${fallback}.`)
  return fallback
}

const NETWORK_HOST = sanitizeNetworkString(MC_HOST, {
  fallback: 'localhost',
  maxLength: 255,
  allowed: /[0-9A-Za-z.\-:]/,
  label: 'host'
})

const NETWORK_PORT = sanitizeNetworkPort(MC_PORT, 25565)

function safeDisconnectReason(reason, fallback = 'shutdown') {
  return sanitizeNetworkString(reason, {
    fallback,
    maxLength: 64,
    allowed: /[0-9A-Za-z _\-.:]/,
    label: 'disconnect reason'
  })
}

function generateRandomName() {
  let attempt = 0
  while (attempt < 25) {
    const prefix = PREFIXES[Math.floor(Math.random() * PREFIXES.length)] ?? ''
    const rock = ROCK_PARTS[Math.floor(Math.random() * ROCK_PARTS.length)] ?? 'Stone'
    const suffix = SUFFIXES[Math.floor(Math.random() * SUFFIXES.length)] ?? 'son'
    const candidate = `${prefix}${rock}${suffix}`
    const sanitized = sanitizeNetworkString(candidate, {
      fallback: '',
      maxLength: MAX_USERNAME_LENGTH,
      allowed: /[0-9A-Za-z_\-]/,
      label: 'username'
    })
    if (sanitized && !usedNames.has(sanitized)) {
      usedNames.add(sanitized)
      return sanitized
    }
    attempt += 1
  }
  let fallbackName = ''
  let safety = 0
  while (!fallbackName || usedNames.has(fallbackName)) {
    if (safety > 50) {
      fallbackName = 'BrainBot'
      break
    }
    fallbackName = sanitizeNetworkString(`BrainBot${Math.floor(Math.random() * 100000)}`, {
      fallback: 'BrainBot',
      maxLength: MAX_USERNAME_LENGTH,
      allowed: /[0-9A-Za-z_\-]/,
      label: 'username'
    })
    safety += 1
  }
  usedNames.add(fallbackName)
  return fallbackName
}

function label(context) {
  return `${context.username}`
}

async function initializeBaselineBrain() {
  try {
    const loaded = await loadBrain(CHECKPOINT_DIR)
    if (loaded) {
      baselineBrain = loaded
    } else {
      baselineBrain = await createBrain(OBS_SIZE, ACTIONS.length)
    }

    if (baselineBrain) {
      baselineBrain.owner = 'hivemind'
    }

    const savedState = await loadBrainState(CHECKPOINT_DIR)
    if (savedState) {
      baselineState = {
        epsilon: typeof savedState.epsilon === 'number' ? savedState.epsilon : EPSILON_START,
        tickCount: typeof savedState.tickCount === 'number' ? savedState.tickCount : 0,
        trainingSteps: typeof savedState.trainingSteps === 'number' ? savedState.trainingSteps : 0,
        cumulativeReward: typeof savedState.cumulativeReward === 'number' ? savedState.cumulativeReward : 0,
        generation: typeof savedState.generation === 'number' ? savedState.generation : 0
      }
      lastSaveTick = baselineState.tickCount
      lastSaveTime = Date.now()
      console.log('[Baseline] Restored checkpoint state.')
    }
  } catch (err) {
    console.error('[Baseline] Failed to initialize from checkpoint:', err)
    baselineBrain = await createBrain(OBS_SIZE, ACTIONS.length)
    if (baselineBrain) {
      baselineBrain.owner = 'hivemind'
    }
  }

  return baselineBrain
}

baselineReady = initializeBaselineBrain()

async function ensureBaselineReady() {
  if (baselineBrain) return baselineBrain
  if (!baselineReady) {
    baselineReady = initializeBaselineBrain()
  }
  try {
    await baselineReady
  } catch (err) {
    console.error('[Baseline] Initialization failed, recreating model:', err)
    baselineBrain = await createBrain(OBS_SIZE, ACTIONS.length)
    if (baselineBrain) {
      baselineBrain.owner = 'hivemind'
    }
    baselineReady = Promise.resolve(baselineBrain)
  }
  return baselineBrain
}

async function persistBaseline(reason = 'periodic') {
  if (!baselineBrain) return
  try {
    await saveBrain(baselineBrain, CHECKPOINT_DIR)
    const avgEpsilon = contexts.length
      ? contexts.reduce((sum, ctx) => sum + ctx.epsilon, 0) / contexts.length
      : baselineState.epsilon
    await saveBrainState(
      {
        epsilon: avgEpsilon,
        tickCount: baselineState.tickCount,
        trainingSteps: baselineState.trainingSteps,
        cumulativeReward: baselineState.cumulativeReward,
        generation: baselineState.generation,
        reason
      },
      CHECKPOINT_DIR
    )
    lastSaveTick = baselineState.tickCount
    lastSaveTime = Date.now()
    console.log(`[Baseline] Saved checkpoint (${reason}).`)
  } catch (err) {
    console.error(`[Baseline] Failed to save checkpoint (${reason}):`, err)
  }
}

function scheduleBaselineSave(reason = 'periodic') {
  if (saveInFlight) {
    pendingSaveReason = reason
    return
  }

  saveInFlight = (async () => {
    await persistBaseline(reason)
  })()

  saveInFlight
    .catch(err => console.error('[Baseline] Save task error:', err))
    .finally(() => {
      saveInFlight = null
      if (pendingSaveReason) {
        const nextReason = pendingSaveReason
        pendingSaveReason = null
        scheduleBaselineSave(nextReason)
      }
    })
}

function maybeTriggerAutosave() {
  const ticksSinceSave = baselineState.tickCount - lastSaveTick
  const msSinceSave = Date.now() - lastSaveTime
  if (ticksSinceSave >= SAVE_INTERVAL_TICKS || msSinceSave >= SAVE_INTERVAL_MS) {
    scheduleBaselineSave('autosave')
  }
}

async function flushPendingSave() {
  if (saveInFlight) {
    try {
      await saveInFlight
    } catch (err) {
      console.error('[Baseline] Pending save failed:', err)
    }
  }
}

async function ensureContextBrain(context) {
  await ensureBaselineReady()
  if (!baselineBrain) {
    throw new Error('Baseline brain failed to initialize')
  }
  if (!context.brain || context.brain.id !== baselineBrain.id) {
    const brainRef = {
      id: baselineBrain.id,
      inputSize: baselineBrain.inputSize,
      actionCount: baselineBrain.actionCount,
      owner: label(context)
    }
    context.brain = brainRef
  }
  return context.brain
}

function gatherObservations(context) {
  const { bot } = context
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

async function equipBestTool(context, preferredKeywords = []) {
  const items = context.bot.inventory?.items?.() ?? []
  for (const keyword of preferredKeywords) {
    const tool = items.find(item => item?.name?.includes(keyword))
    if (tool) {
      try {
        await context.bot.equip(tool, 'hand')
        return true
      } catch (err) {
        console.warn(`[${label(context)}] Failed to equip ${tool.name}:`, err?.message ?? err)
      }
    }
  }
  return false
}

async function equipPlaceableBlock(context) {
  const items = context.bot.inventory?.items?.() ?? []
  for (const item of items) {
    if (!item?.name) continue
    if (['sword', 'pickaxe', 'axe', 'shovel', 'hoe', 'bucket'].some(tool => item.name.includes(tool))) continue
    try {
      await context.bot.equip(item, 'hand')
      return true
    } catch (err) {
      console.warn(`[${label(context)}] Failed to equip ${item.name} for building:`, err?.message ?? err)
    }
  }
  return false
}

function findNearbyBlock(context, name, maxDistance = 4) {
  if (!context.registry) return null
  const blockInfo = context.registry.blocksByName?.[name]
  if (!blockInfo) return null
  try {
    return context.bot.findBlock({ matching: blockInfo.id, maxDistance }) ?? null
  } catch (err) {
    console.warn(`[${label(context)}] findNearbyBlock failed for ${name}:`, err?.message ?? err)
    return null
  }
}

async function craftItem(context, targetName, options = {}) {
  if (!context.registry) return false

  const {
    amount = 1,
    requireTable = false,
    allowPartial = true,
    tableRange = 4
  } = options

  const itemInfo = context.registry.itemsByName?.[targetName]
  if (!itemInfo) {
    console.warn(`[${label(context)}] Unknown craft target: ${targetName}`)
    return false
  }

  const tableBlock = findNearbyBlock(context, 'crafting_table', tableRange)
  const candidates = []

  if (tableBlock) {
    candidates.push(tableBlock)
  }
  if (!requireTable || !tableBlock) {
    candidates.push(null)
  }

  if (requireTable && !tableBlock) {
    console.log(`[${label(context)}] Crafting table required but not nearby.`)
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
        const recipes = context.bot.recipesFor(itemInfo.id, null, craftCount, table ?? null)
        if (recipes?.length) {
          try {
            await context.bot.craft(recipes[0], craftCount, table ?? undefined)
            return true
          } catch (err) {
            console.warn(`[${label(context)}] Craft ${targetName} x${craftCount} failed:`, err?.message ?? err)
          }
        }
      } catch (err) {
        console.warn(`[${label(context)}] recipesFor failed for ${targetName}:`, err?.message ?? err)
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

async function executeCraftAction(context, act) {
  const config = CRAFTING_ACTIONS[act]
  if (!config) return

  const success = await craftItem(context, config.item, config)
  if (success) {
    context.blockReward += config.reward ?? 0.4
    console.log(`[${label(context)}] Crafted ${config.item}`)
  } else {
    context.blockReward -= 0.03
  }
}

async function executeAction(context, index) {
  const act = ACTIONS[index]
  if (!act) return
  console.log(`[${label(context)}] Executing: ${act}`)

  const { bot } = context

  if (CRAFTING_ACTIONS[act]) {
    await executeCraftAction(context, act)
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
        await holdControls(['forward', 'jump'])
        break
      case 'sprint_forward':
        await holdControls(['forward', 'sprint'], 500)
        break
      case 'sneak_forward':
        await holdControls(['forward', 'sneak'], 500)
        break
      case 'turn_left':
        await lookBy(-Math.PI / 4, 0)
        break
      case 'turn_right':
        await lookBy(Math.PI / 4, 0)
        break
      case 'look_up':
        await lookBy(0, -Math.PI / 8)
        break
      case 'look_down':
        await lookBy(0, Math.PI / 8)
        break
      case 'mine': {
        const target = getTargetBlock()
        if (target) {
          const mined = await equipBestTool(context, ['pickaxe', 'axe', 'shovel'])
          if (!mined) {
            await equipBestTool(context, ['hand'])
          }
          try {
            await bot.dig(target)
            context.blockReward += 0.5
          } catch (err) {
            console.warn(`[${label(context)}] Mining failed:`, err?.message ?? err)
            context.blockReward -= 0.05
          }
        } else {
          context.blockReward -= 0.02
        }
        break
      }
      case 'attack': {
        const entity = bot.nearestEntity()
        if (entity) {
          try {
            await bot.attack(entity)
            context.blockReward += 0.3
          } catch (err) {
            console.warn(`[${label(context)}] Attack failed:`, err?.message ?? err)
            context.blockReward -= 0.05
          }
        } else {
          context.blockReward -= 0.01
        }
        break
      }
      case 'build': {
        const target = getTargetBlock()
        if (target) {
          const placePos = target.position.offset(0, 1, 0)
          const success = await equipPlaceableBlock(context)
          if (success) {
            try {
              await bot.placeBlock(target, new Vec3(0, 1, 0))
              context.blockReward += 0.25
            } catch (err) {
              console.warn(`[${label(context)}] Build failed:`, err?.message ?? err)
              context.blockReward -= 0.02
            }
          } else {
            context.blockReward -= 0.02
          }
          if (placePos) {
            // noop - placeholder for potential future heuristics
          }
        } else {
          context.blockReward -= 0.02
        }
        break
      }
      case 'build_above': {
        const success = await equipPlaceableBlock(context)
        if (success) {
          const eyePos = bot.entity?.position
          if (eyePos) {
            const targetPos = eyePos.offset(0, 1, 0)
            const blockBelow = bot.blockAt(targetPos.offset(0, -1, 0))
            if (blockBelow) {
              try {
                await bot.placeBlock(blockBelow, new Vec3(0, 1, 0))
                context.blockReward += 0.2
              } catch (err) {
                console.warn(`[${label(context)}] Build above failed:`, err?.message ?? err)
                context.blockReward -= 0.02
              }
            }
          }
        } else {
          context.blockReward -= 0.02
        }
        break
      }
      case 'use_item': {
        try {
          bot.activateItem()
          await sleep(300)
          bot.deactivateItem()
          context.blockReward += 0.05
        } catch (err) {
          console.warn(`[${label(context)}] Use-item action failed:`, err?.message ?? err)
        }
        break
      }
      default:
        break
    }
  } finally {
    bot.clearControlStates()
  }
}

function computeReward(context, obs) {
  let reward = 0

  const pos = { x: obs[0], y: obs[1], z: obs[2] }
  if (context.lastPos) {
    const dx = pos.x - context.lastPos.x
    const dy = pos.y - context.lastPos.y
    const dz = pos.z - context.lastPos.z
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz)
    const horizontal = Math.sqrt(dx * dx + dz * dz)
    reward += Math.min(dist * 0.1, 0.5)
    reward += Math.min(horizontal * 0.05, 0.25)
    reward += Math.min(Math.abs(dy) * 0.05, 0.15)
  }

  const health = obs[8]
  if (Number.isFinite(health)) {
    if (health < context.lastHealth) {
      reward -= Math.min(1, (context.lastHealth - health) * 0.5)
    } else if (health > context.lastHealth) {
      reward += Math.min(1, (health - context.lastHealth) * 0.5)
    }
    context.lastHealth = health
  }

  const food = obs[9]
  if (Number.isFinite(food)) {
    if (food > context.lastFood) {
      reward += Math.min(0.5, (food - context.lastFood) * 0.1)
    } else if (food < context.lastFood) {
      reward -= Math.min(0.5, (context.lastFood - food) * 0.05)
    }
    context.lastFood = food
  }

  const invTotal = obs[18]
  if (Number.isFinite(invTotal)) {
    const delta = invTotal - context.lastInvTotal
    if (delta !== 0) {
      reward += Math.sign(delta) * Math.min(Math.abs(delta) * 0.2, 1.5)
    }
    context.lastInvTotal = invTotal
  }

  const nearestDist = obs[16]
  if (Number.isFinite(nearestDist) && nearestDist > 0 && nearestDist < 3) {
    reward -= (3 - nearestDist) * 0.05
  }

  const consumedBlockReward = context.blockReward
  reward += consumedBlockReward
  context.blockReward = 0

  reward -= 0.02

  context.lastPos = { ...pos }
  return reward
}

async function tickLoop(context) {
  if (!globalRunning || !context.running || context.tickInFlight) return
  context.tickInFlight = true

  try {
    await ensureContextBrain(context)
    if (!context.bot?.entity?.position) {
      console.warn(`[${label(context)}] Entity not ready, skipping tick.`)
      return
    }

    if (!context.registry && context.bot.registry) {
      context.registry = context.bot.registry
    }

    const observation = gatherObservations(context)
    if (Array.isArray(context.lastPrediction) && context.lastPrediction.length === observation.length) {
      let mse = 0
      for (let i = 0; i < observation.length; i++) {
        const diff = (observation[i] ?? 0) - (context.lastPrediction[i] ?? 0)
        mse += diff * diff
      }
      context.lastPredictionError = Math.sqrt(mse / observation.length)
    } else {
      context.lastPredictionError = null
    }
    const reward = computeReward(context, observation)

    let trained = false
    if (context.lastObs && context.lastAction != null) {
      trained = await trainBrain(
        context.brain,
        context.lastObs,
        context.lastAction,
        reward,
        observation
      )
    }

    const { action, prediction } = await chooseAction(context.brain, observation, context.epsilon)
    await executeAction(context, action)

    context.lastObs = observation
    context.lastAction = action
    context.lastPrediction = Array.isArray(prediction) ? prediction : null

    context.tickCount += 1
    context.generationTicks += 1
    context.cumulativeReward += reward
    context.generationReward += reward
    baselineState.tickCount += 1
    baselineState.cumulativeReward += reward
    if (trained) {
      context.trainingSteps += 1
      baselineState.trainingSteps += 1
    }

    if (context.epsilon > EPSILON_MIN) {
      context.epsilon = Math.max(EPSILON_MIN, context.epsilon * EPSILON_DECAY)
    }

    const predictionNote = context.lastPredictionError != null
      ? ` | PredErr: ${context.lastPredictionError.toFixed(3)}`
      : ''
    console.log(`[${label(context)}] Tick done | Reward: ${reward.toFixed(3)} | Eps: ${context.epsilon.toFixed(3)}${predictionNote}`)
    maybeTriggerAutosave()
    await maybeCompleteGeneration(context)
  } catch (err) {
    console.error(`[${label(context)}] Tick error:`, err)
  } finally {
    context.tickInFlight = false

    if (globalRunning && context.running) {
      context.tickTimer = setTimeout(() => {
        context.tickTimer = null
        tickLoop(context).catch(err => console.error(`[${label(context)}] Tick scheduling error:`, err))
      }, TICK_RATE)
    }
  }
}

async function maybeCompleteGeneration(context) {
  if (context.generationTicks < GENERATION_TICKS) {
    return
  }

  context.readyForSync = true
  console.log(`[${label(context)}] Completed generation window with reward ${context.generationReward.toFixed(2)}.`)
  await synchronizeGeneration()
}

async function synchronizeGeneration() {
  if (generationSyncInFlight) return
  if (!contexts.length) return
  if (!contexts.every(ctx => ctx.readyForSync)) return

  generationSyncInFlight = true
  try {
    const sorted = [...contexts].sort((a, b) => b.generationReward - a.generationReward)
    const topTwo = sorted.slice(0, 2)
    await ensureBaselineReady()

    baselineState.generation += 1
    const leaderboard = topTwo.length
      ? topTwo.map(ctx => `${label(ctx)}=${ctx.generationReward.toFixed(2)}`).join(', ')
      : 'n/a'
    console.log(`[Baseline] Generation ${baselineState.generation} | Top rewards: ${leaderboard}`)

    for (const ctx of contexts) {
      ctx.generationTicks = 0
      ctx.generationReward = 0
      ctx.readyForSync = false
    }

    scheduleBaselineSave('generation')
  } catch (err) {
    console.error('[Baseline] Failed to synchronize generation:', err)
  } finally {
    generationSyncInFlight = false
  }
}

function setupRewardTracking(context) {
  const { bot } = context

  bot.on('blockBreak', block => {
    if (!block || block.name === 'air') return
    const value =
      block.name.includes('ore') ? 2.0 :
      block.name.includes('stone') ? 1.0 :
      block.name.includes('dirt') ? 0.5 :
      0.3
    context.blockReward += value
    console.log(`[${label(context)}] Broke ${block.name} → +${value.toFixed(2)} reward`)
  })

  bot.on('diggingAborted', () => {
    context.blockReward -= 0.1
    console.log(`[${label(context)}] Dig aborted → -0.1 penalty`)
  })

  bot.on('playerCollect', (collector, collected) => {
    if (collector === bot.entity) {
      const count = collected?.metadata?.itemCount ?? collected?.count ?? 1
      const bonus = Math.max(0.3, (count || 1) * 0.15)
      context.blockReward += bonus
      console.log(`[${label(context)}] Collected item → +${bonus.toFixed(2)} reward`)
    }
  })
}

function setupBot(context) {
  const username = context.username
  console.log(`[${username}] Connecting to ${NETWORK_HOST}:${NETWORK_PORT}`)

  context.bot.once('spawn', () => {
    console.log(`[${label(context)}] Spawned! Waiting for entity to initialize...`)

    context.registry = context.bot.registry ?? context.registry
    if (!context.registry) {
      console.warn(`[${label(context)}] Failed to load registry — crafting actions will be limited.`)
    }

    const waitForEntity = setInterval(() => {
      if (context.bot?.entity?.position) {
        clearInterval(waitForEntity)
        console.log(`[${label(context)}] Entity ready — starting tick loop!`)

        tickLoop(context).catch(err => console.error(`[${label(context)}] Initial tick error:`, err))
      }
    }, 500)
  })

  context.bot.on('error', e => console.error(`[${label(context)}] Error:`, e))
  context.bot.on('kicked', r => console.error(`[${label(context)}] Kicked:`, r))

  setupRewardTracking(context)
}

function createContext(index) {
  const username = generateRandomName()
  const bot = mineflayer.createBot({
    host: NETWORK_HOST,
    port: NETWORK_PORT,
    username
  })

  const context = {
    id: index,
    username,
    bot,
    brain: null,
    epsilon: EPSILON_START,
    running: true,
    tickTimer: null,
    tickInFlight: false,
    registry: null,
    lastObs: null,
    lastAction: null,
    lastPrediction: null,
    lastPredictionError: null,
    lastPos: null,
    lastHealth: 20,
    lastFood: 20,
    lastInvTotal: 0,
    blockReward: 0,
    tickCount: 0,
    trainingSteps: 0,
    cumulativeReward: 0,
    generationTicks: 0,
    generationReward: 0,
    readyForSync: false
  }

  setupBot(context)
  contexts.push(context)
  return context
}

for (let i = 0; i < BOT_COUNT; i++) {
  createContext(i)
}

process.stdin.resume()
process.stdin.setEncoding('utf8')
console.log('[Brain] Type "pause", "resume", "save", or "exit".')

process.stdin.on('data', async data => {
  const cmd = data.trim().toLowerCase()
  if (cmd === 'pause') {
    globalRunning = false
    for (const ctx of contexts) {
      ctx.running = false
      if (ctx.tickTimer) {
        clearTimeout(ctx.tickTimer)
        ctx.tickTimer = null
      }
    }
    console.log('[Brain] Paused. Current ticks will finish before stopping.')
  } else if (cmd === 'resume') {
    if (!globalRunning) {
      globalRunning = true
      for (const ctx of contexts) {
        if (!ctx.running) {
          ctx.running = true
          if (!ctx.tickInFlight && !ctx.tickTimer) {
            tickLoop(ctx).catch(err => console.error(`[${label(ctx)}] Resume tick error:`, err))
          }
        }
      }
      console.log('[Brain] Resumed.')
    }
  } else if (cmd === 'save') {
    console.log('[Brain] Manual save requested...')
    scheduleBaselineSave('manual')
  } else if (['exit', 'quit', 'stop'].includes(cmd)) {
    console.log('[Brain] Saving model + shutting down...')
    globalRunning = false
    for (const ctx of contexts) {
      ctx.running = false
      if (ctx.tickTimer) {
        clearTimeout(ctx.tickTimer)
        ctx.tickTimer = null
      }
    }
    await ensureBaselineReady()
    await flushPendingSave()
    await persistBaseline('shutdown')
    for (const ctx of contexts) {
      try {
        ctx.bot.quit(safeDisconnectReason('Manual shutdown'))
      } catch (err) {
        console.warn(`[${label(ctx)}] Failed to quit bot during shutdown:`, err)
      }
    }
    process.exit(0)
  }
})

async function gracefulShutdown(reason = 'signal') {
  try {
    console.log(`[Brain] Caught ${reason}. Saving before exit...`)
    globalRunning = false
    for (const ctx of contexts) {
      ctx.running = false
      if (ctx.tickTimer) {
        clearTimeout(ctx.tickTimer)
        ctx.tickTimer = null
      }
    }
    await ensureBaselineReady()
    await flushPendingSave()
    await persistBaseline(reason)
    for (const ctx of contexts) {
      try {
        ctx.bot.quit(safeDisconnectReason(`Shutdown: ${reason}`))
      } catch (err) {
        console.warn(`[${label(ctx)}] Failed to quit bot during shutdown:`, err)
      }
    }
  } catch (err) {
    console.error('[Brain] Failed during graceful shutdown:', err)
  } finally {
    process.exit(0)
  }
}

process.on('SIGINT', () => gracefulShutdown('SIGINT'))
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'))
