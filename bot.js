import mineflayer from 'mineflayer'
import { Vec3 } from 'vec3'
import fs from 'node:fs'
import path from 'node:path'
import { pipeline } from 'node:stream'
import { promisify } from 'node:util'
import zlib from 'node:zlib'
import { performance } from 'node:perf_hooks'
import {
  createBrain,
  saveBrain,
  loadBrain,
  saveBrainState,
  loadBrainState,
  DEFAULT_BRAIN_DIR,
  mutateWeights,
  averageWeights,
  copyWeights,
  isRemoteBrainConnected,
  getRemoteBrainStatus,
  RemoteBrainUnavailableError
} from './brainClient.js'
import {
  chooseActionConcurrent,
  trainBrainConcurrent,
  shutdownBrainWorkerPool,
  warmBrainWorkerPool
} from './brainWorkerPool.js'
import {
  isEchoFallbackEnabled,
  chooseEchoAction,
  reportEchoLearning
} from './echoTool.js'

// ----------------------------
// CONFIG
// ----------------------------
const MC_HOST = 'localhost'
const MC_PORT = 25565
const BOT_COUNT = Math.max(2, parseInt(process.env.BOT_COUNT ?? '10', 10))
const MIN_BOTS = Math.max(2, parseInt(process.env.BOT_MIN ?? '2', 10))
const GENERATION_TICKS = Math.max(50, parseInt(process.env.GENERATION_TICKS ?? '200', 10))
const ROCK_PARTS = ['Rock', 'Stone', 'Grav', 'Ore', 'Pebble', 'Granite', 'Basalt', 'Iron', 'Coal', 'Quartz']
const SUFFIXES = ['son', 'grip', 'deep', 'delver', 'breaker', 'forge', 'drill', 'hammer', 'core', 'blast']
const PREFIXES = ['', 'Mc', 'Von', 'De', "O'", 'El']
const LINEAGE_VARIANT_SUFFIXES = ['Nova', 'Flux', 'Echo', 'Shard', 'Pulse', 'Varia', 'Drift', 'Spark']

const LINEAGE_ROOT_NAME = 'VonBasaltdeep'
const FERAL_DEFAULT_RATIO = Math.min(0.5, Math.max(0, Number.parseFloat(process.env.FERAL_RATIO ?? '0.2')))
const FERAL_INTERVAL = Math.max(2, Math.round(1 / (FERAL_DEFAULT_RATIO || 0.2)))
const EMOTION_DECAY = 0.92
const MORALE_BASELINE = 0.55
const REWARD_MUTATION_INTERVAL = Math.max(500, parseInt(process.env.REWARD_MUTATION_INTERVAL ?? '2500', 10))
const REWARD_MUTATION_JITTER = Math.max(100, parseInt(process.env.REWARD_MUTATION_JITTER ?? '600', 10))
const CROSSOVER_GENERATION_INTERVAL = Math.max(1, parseInt(process.env.CROSSOVER_INTERVAL ?? '5', 10))

const LINEAGE_DOMINANCE_SHARE_THRESHOLD = readNumberEnv('BOT_LINEAGE_DOMINANCE_SHARE', 0.62, {
  min: 0.3,
  max: 0.95
})
const LINEAGE_DOMINANCE_GAP_THRESHOLD = readNumberEnv('BOT_LINEAGE_DOMINANCE_GAP', 0.18, {
  min: 0,
  max: 0.6
})
const LINEAGE_DOMINANCE_STREAK_THRESHOLD = Math.max(
  1,
  Math.floor(readNumberEnv('BOT_LINEAGE_DOMINANCE_STREAK', 2, { min: 1, max: 10 }))
)
const LINEAGE_DOMINANCE_MUTATION_MULTIPLIER = readNumberEnv(
  'BOT_LINEAGE_DOMINANCE_MUTATION',
  1.35,
  { min: 0.1, max: 5 }
)
const LINEAGE_DOMINANCE_EXTRA_MUTATIONS = Math.max(
  1,
  Math.floor(readNumberEnv('BOT_LINEAGE_DOMINANCE_EXTRA_MUTATIONS', 2, { min: 1, max: 6 }))
)
const LINEAGE_DOMINANCE_COOLDOWN_GENERATIONS = Math.max(
  1,
  Math.floor(readNumberEnv('BOT_LINEAGE_DOMINANCE_COOLDOWN', 3, { min: 1, max: 12 }))
)

const NON_FINITE_STRIKE_WINDOW_MS = Math.max(
  1000,
  Math.floor(readNumberEnv('BOT_NON_FINITE_STRIKE_WINDOW_MS', 120000, { min: 1000, max: 3600000 }))
)
const NON_FINITE_REASON_THRESHOLD = Math.max(
  2,
  Math.floor(readNumberEnv('BOT_NON_FINITE_REASON_THRESHOLD', 3, { min: 1, max: 10 }))
)
const NON_FINITE_TOTAL_THRESHOLD = Math.max(
  NON_FINITE_REASON_THRESHOLD,
  Math.floor(readNumberEnv('BOT_NON_FINITE_TOTAL_THRESHOLD', 4, { min: 2, max: 16 }))
)
const NON_FINITE_MAX_ESCALATIONS = Math.max(
  1,
  Math.floor(readNumberEnv('BOT_NON_FINITE_MAX_ESCALATIONS', 2, { min: 1, max: 6 }))
)
const NON_FINITE_MUTATION_MULTIPLIER = readNumberEnv('BOT_NON_FINITE_MUTATION_MULTIPLIER', 0.75, {
  min: 0.05,
  max: 4
})
const NON_FINITE_MUTATION_MIN = readNumberEnv('BOT_NON_FINITE_MUTATION_MIN', 0.05, {
  min: 0,
  max: 2
})
const NON_FINITE_EPSILON_BOOST = readNumberEnv('BOT_NON_FINITE_EPSILON_BOOST', 0.35, {
  min: 0,
  max: 0.95
})

const DEFAULT_MINING_TOOL_PREFERENCES = ['pickaxe', 'axe', 'shovel']
const TOOL_TIER_WEIGHTS = [
  { keyword: 'netherite', score: 120 },
  { keyword: 'diamond', score: 100 },
  { keyword: 'golden', score: 90 },
  { keyword: 'gold', score: 85 },
  { keyword: 'iron', score: 75 },
  { keyword: 'stone', score: 60 },
  { keyword: 'wooden', score: 45 },
  { keyword: 'wood', score: 45 }
]
const TOOL_TYPE_WEIGHTS = {
  pickaxe: 80,
  axe: 65,
  shovel: 55,
  hoe: 20,
  shears: 30
}

const FATAL_RECOVERY_COOLDOWN_MS = Math.max(
  5000,
  Math.floor(readNumberEnv('BOT_RECOVERY_COOLDOWN_MS', 15000, { min: 1000 }))
)
const FATAL_RECOVERY_RESTART_DELAY_MS = Math.max(
  2000,
  Math.floor(readNumberEnv('BOT_RECOVERY_RESTART_DELAY_MS', 5000, { min: 1000 }))
)
const FATAL_RECOVERY_MAX_ATTEMPTS = Math.max(
  1,
  Math.floor(readNumberEnv('BOT_RECOVERY_MAX_ATTEMPTS', 5, { min: 1 }))
)

const HUNGER_EAT_THRESHOLD = Math.max(12, parseInt(process.env.BOT_AUTOEAT_THRESHOLD ?? '14', 10))
const HUNGER_CRITICAL_THRESHOLD = Math.max(6, Math.min(HUNGER_EAT_THRESHOLD - 2, parseInt(process.env.BOT_HUNGER_CRITICAL ?? '8', 10)))
const AUTO_EAT_CHECK_INTERVAL_MS = 2500
const ARMOR_CHECK_INTERVAL_MS = 4000
const HUNGER_HUNT_THRESHOLD = Math.max(8, Math.min(HUNGER_EAT_THRESHOLD, parseInt(process.env.BOT_HUNT_HUNGER_THRESHOLD ?? '12', 10)))
const HUNGER_HUNT_ACTION_REWARD = 0.35
const HUNGER_COLLECTION_REWARD = 0.25

const ORE_NAME_KEYWORDS = [
  'ore',
  'ingot',
  'debris',
  'gem',
  'emerald',
  'lapis',
  'quartz',
  'diamond',
  'netherite',
  'coal',
  'redstone'
]
const RAW_ORE_SUFFIXES = new Set([
  'iron',
  'gold',
  'copper'
])
const ORE_BLOCK_REWARD_BONUS = readNumberEnv('ORE_BLOCK_REWARD_BONUS', 6, {
  min: 0,
  max: 100
})
const ORE_PICKUP_REWARD_BASE = readNumberEnv('ORE_PICKUP_REWARD_BASE', 3, {
  min: 0,
  max: 100
})
const ORE_PICKUP_REWARD_PER_ITEM = readNumberEnv('ORE_PICKUP_REWARD_PER_ITEM', 0.75, {
  min: 0,
  max: 20
})

const STRAIGHT_DOWN_DIG_THRESHOLD = Math.max(
  2,
  Math.floor(readNumberEnv('STRAIGHT_DOWN_DIG_THRESHOLD', 3, { min: 1, max: 20 }))
)
const STRAIGHT_DOWN_DIG_PENALTY = readNumberEnv('STRAIGHT_DOWN_DIG_PENALTY', 1.25, {
  min: 0,
  max: 20
})
const STRAIGHT_DOWN_DIG_PENALTY_GROWTH = readNumberEnv('STRAIGHT_DOWN_DIG_PENALTY_GROWTH', 0.4, {
  min: 0,
  max: 10
})

const COOKED_FOOD_KEYWORDS = ['cooked', 'baked', 'roasted', 'stew', 'pie', 'bread']
const AVOID_FOOD_KEYWORDS = ['rotten_flesh', 'spider_eye', 'poisonous', 'raw_fish', 'raw_salmon']
const PASSIVE_ANIMAL_KEYWORDS = ['cow', 'pig', 'sheep', 'chicken', 'rabbit', 'mooshroom', 'goat', 'hoglin', 'salmon', 'cod']

const DEATH_REWARD_PENALTY = (() => {
  const raw = Number.parseFloat(process.env.BOT_DEATH_REWARD_PENALTY ?? '20')
  if (!Number.isFinite(raw) || raw <= 0) {
    return 20
  }
  return Math.max(5, raw)
})()

const COSINE_PENALTY_REFERENCE = readNumberEnv('COSINE_PENALTY_REFERENCE', 3.5, {
  min: 0.1,
  max: 100
})
const COSINE_PENALTY_DECAY = readNumberEnv('COSINE_PENALTY_DECAY', 0.9, {
  min: 0,
  max: 0.999
})
const COSINE_PENALTY_MIN_SCALE = readNumberEnv('COSINE_PENALTY_MIN_SCALE', 0.35, {
  min: 0,
  max: 1
})
const COSINE_PENALTY_MAX_SCALE_RAW = readNumberEnv('COSINE_PENALTY_MAX_SCALE', 1.0, {
  min: 0.1,
  max: 2.5
})
const COSINE_PENALTY_MAX_SCALE = Math.max(COSINE_PENALTY_MIN_SCALE, COSINE_PENALTY_MAX_SCALE_RAW)

const ARMOR_SLOT_PATTERNS = [
  { slot: 'head', keywords: ['helmet', 'cap', 'turtle_helmet', 'turtle_shell'] },
  { slot: 'torso', keywords: ['chestplate', 'tunic', 'elytra'] },
  { slot: 'legs', keywords: ['leggings', 'pants'] },
  { slot: 'feet', keywords: ['boots'] }
]

const ARMOR_TIER_WEIGHTS = [
  { keyword: 'netherite', score: 160 },
  { keyword: 'diamond', score: 130 },
  { keyword: 'golden', score: 110 },
  { keyword: 'gold', score: 105 },
  { keyword: 'iron', score: 95 },
  { keyword: 'chainmail', score: 85 },
  { keyword: 'stone', score: 60 },
  { keyword: 'leather', score: 55 }
]

const ARMOR_SLOT_BONUS = {
  head: 25,
  torso: 35,
  legs: 30,
  feet: 20
}

const REWARD_SIGN = Object.freeze({
  POSITIVE: 'positive',
  NEGATIVE: 'negative',
  EITHER: 'either'
})

function readNumberEnv(name, fallback, { min = -Infinity, max = Infinity } = {}) {
  const raw = Number.parseFloat(process.env[name] ?? '')
  if (!Number.isFinite(raw)) {
    return fallback
  }
  return Math.min(max, Math.max(min, raw))
}

function recordRewardSignCorrection(context, expectation, original, corrected, reason, channel) {
  if (!context || corrected === original) return
  if (!context.rewardSignStats) {
    context.rewardSignStats = { corrections: 0, history: [] }
  }
  context.rewardSignStats.corrections += 1
  const entry = {
    expectation,
    original,
    corrected,
    reason,
    channel,
    tick: context.tickCount ?? 0
  }
  context.rewardSignStats.history.push(entry)
  if (context.rewardSignStats.history.length > 8) {
    context.rewardSignStats.history.shift()
  }
  const shouldLog =
    context.rewardSignStats.corrections <= 5 || context.rewardSignStats.corrections % 20 === 0
  if (shouldLog) {
    try {
      console.warn(
        `[${label(context)}] Reward sign correction for ${reason} (${channel}) → ${original.toFixed(3)} adjusted to ${corrected.toFixed(3)} (expected ${expectation}).`
      )
    } catch (err) {
      console.warn('Reward sign correction logged without context label:', err)
    }
  }
}

function ensureRewardSign(amount, expectation = REWARD_SIGN.EITHER, context = null, reason = 'unspecified', channel = 'reward') {
  const numeric = Number(amount)
  if (!Number.isFinite(numeric) || numeric === 0) {
    return 0
  }
  let corrected = numeric
  if (expectation === REWARD_SIGN.POSITIVE && numeric < 0) {
    corrected = Math.abs(numeric)
  } else if (expectation === REWARD_SIGN.NEGATIVE && numeric > 0) {
    corrected = -Math.abs(numeric)
  }
  if (corrected !== numeric) {
    recordRewardSignCorrection(context, expectation, numeric, corrected, reason, channel)
  }
  return corrected
}

function createRewardAccumulator() {
  return { total: 0, reward: 0, penalty: 0 }
}

function ensureRewardAccumulator(value) {
  if (
    value &&
    typeof value === 'object' &&
    Object.prototype.hasOwnProperty.call(value, 'total') &&
    Object.prototype.hasOwnProperty.call(value, 'reward') &&
    Object.prototype.hasOwnProperty.call(value, 'penalty')
  ) {
    const accumulator = value
    accumulator.total = Number.isFinite(accumulator.total) ? accumulator.total : 0
    accumulator.reward = Number.isFinite(accumulator.reward) && accumulator.reward > 0
      ? accumulator.reward
      : 0
    accumulator.penalty = Number.isFinite(accumulator.penalty) && accumulator.penalty > 0
      ? accumulator.penalty
      : 0
    return accumulator
  }
  const numeric = Number(value)
  if (!Number.isFinite(numeric) || numeric === 0) {
    return createRewardAccumulator()
  }
  if (numeric > 0) {
    return { total: numeric, reward: numeric, penalty: 0 }
  }
  const magnitude = Math.abs(numeric)
  return { total: numeric, reward: 0, penalty: magnitude }
}

function finalizeRewardAccumulator(value) {
  const accumulator = ensureRewardAccumulator(value)
  return {
    total: clampReward(accumulator.total),
    reward: limitPositive(accumulator.reward, MAX_REWARD_MAGNITUDE),
    penalty: limitPositive(accumulator.penalty, MAX_REWARD_MAGNITUDE)
  }
}

function applyRewardComponent(total, amount, expectation = REWARD_SIGN.EITHER, context = null, reason = 'unspecified') {
  const accumulator = ensureRewardAccumulator(total)
  const corrected = ensureRewardSign(amount, expectation, context, reason, 'reward')
  const adjusted = applyCosinePenaltyScaling(context, reason, corrected)
  if (!Number.isFinite(adjusted) || adjusted === 0) {
    return accumulator
  }
  accumulator.total += adjusted
  if (adjusted > 0) {
    accumulator.reward += adjusted
  } else if (adjusted < 0) {
    accumulator.penalty += Math.abs(adjusted)
  }
  accumulator.reward = Math.max(0, accumulator.reward)
  accumulator.penalty = Math.max(0, accumulator.penalty)
  return accumulator
}

function addBlockReward(context, amount, expectation = REWARD_SIGN.EITHER, reason = 'block') {
  if (!context) return 0
  const base = Number.isFinite(context.blockReward) ? context.blockReward : 0
  const corrected = ensureRewardSign(amount, expectation, context, reason, 'block')
  const adjusted = applyCosinePenaltyScaling(context, reason, corrected)
  if (adjusted === 0) {
    context.blockReward = base
    return context.blockReward
  }
  const next = base + adjusted
  context.blockReward = Number.isFinite(next) ? next : 0
  return context.blockReward
}

function drainPositiveBlockReward(context, amount) {
  if (!context) return 0
  const deduction = Math.abs(Number(amount) || 0)
  if (!Number.isFinite(deduction) || deduction === 0) {
    return Number.isFinite(context.blockReward) ? context.blockReward : 0
  }
  const base = Number.isFinite(context.blockReward) ? context.blockReward : 0
  if (base <= 0) {
    context.blockReward = base
    return base
  }
  const next = Math.max(0, base - deduction)
  context.blockReward = next
  return next
}

function ensureCosinePenaltyState(context) {
  if (!context) return null
  if (!context.cosinePenaltyScaling) {
    context.cosinePenaltyScaling = {
      reasons: Object.create(null)
    }
  }
  if (!context.cosinePenaltyScaling.reasons) {
    context.cosinePenaltyScaling.reasons = Object.create(null)
  }
  return context.cosinePenaltyScaling.reasons
}

function computeCosinePenaltyScale(context, reason, penaltyValue) {
  if (!context || !Number.isFinite(penaltyValue)) {
    return 1
  }
  const reasons = ensureCosinePenaltyState(context)
  if (!reasons) return 1
  const key = reason || 'unspecified'
  const now = Number.isFinite(context.tickCount) ? context.tickCount : 0
  let state = reasons[key]
  if (!state) {
    state = { phase: 0, lastTick: now }
    reasons[key] = state
  }
  const elapsed = Math.max(0, now - (Number.isFinite(state.lastTick) ? state.lastTick : now))
  const decay = elapsed > 0 ? Math.pow(COSINE_PENALTY_DECAY, elapsed) : 1
  const previousPhase = Number.isFinite(state.phase) ? state.phase : 0
  let phase = Math.max(0, Math.min(1, previousPhase * decay))
  const severity = Math.min(1, Math.abs(penaltyValue) / COSINE_PENALTY_REFERENCE)
  phase = Math.max(0, Math.min(1, phase + severity * (1 - phase)))
  state.phase = phase
  state.lastTick = now
  const cosine = 0.5 * (1 - Math.cos(Math.PI * phase))
  const scale = COSINE_PENALTY_MIN_SCALE + cosine * (COSINE_PENALTY_MAX_SCALE - COSINE_PENALTY_MIN_SCALE)
  return Number.isFinite(scale) && scale > 0 ? scale : 1
}

function applyCosinePenaltyScaling(context, reason, value) {
  if (!context || !Number.isFinite(value) || value >= 0) {
    return Number.isFinite(value) ? value : 0
  }
  const scale = computeCosinePenaltyScale(context, reason, value)
  const adjusted = value * scale
  if (!Number.isFinite(adjusted)) {
    return value
  }
  return adjusted
}

const MOVEMENT_SMOOTH_RAMP_MS = Math.max(
  10,
  Math.floor(readNumberEnv('MOVEMENT_SMOOTH_RAMP_MS', 120, { min: 10 }))
)
const MOVEMENT_SMOOTH_THRESHOLD = readNumberEnv('MOVEMENT_SMOOTH_THRESHOLD', 0.35, {
  min: 0.05,
  max: 0.9
})
const MOVEMENT_UPDATE_INTERVAL_MS = Math.max(
  10,
  Math.floor(readNumberEnv('MOVEMENT_UPDATE_INTERVAL_MS', 50, { min: 10 }))
)
const MOVEMENT_JUMP_MAX_MS = Math.max(
  50,
  Math.floor(readNumberEnv('MOVEMENT_JUMP_MAX_MS', 300, { min: 50 }))
)
const LOOK_SMOOTH_STEPS = Math.max(
  1,
  Math.floor(readNumberEnv('LOOK_SMOOTH_STEPS', 6, { min: 1, max: 24 }))
)
const LOOK_SMOOTH_DURATION_MS = Math.max(
  20,
  Math.floor(readNumberEnv('LOOK_SMOOTH_DURATION_MS', 180, { min: 20 }))
)

const HEALTH_METRIC_WINDOW_MS = Math.max(
  1000,
  Math.floor(readNumberEnv('HEALTH_METRIC_WINDOW_MS', 300000, { min: 1000 }))
)
const HEALTH_SUMMARY_INTERVAL_MS = Math.max(
  5000,
  Math.floor(readNumberEnv('HEALTH_SUMMARY_INTERVAL_MS', 60000, { min: 5000 }))
)

const TEMPLATE_TOP_BRAINS = Math.max(1, Math.floor(readNumberEnv('TOP_TEMPLATE_BRAINS', 3, { min: 1 })))
const OLDEST_RETIRE_PER_GENERATION = Math.max(
  1,
  Math.floor(readNumberEnv('OLDEST_RETIRE_PER_GENERATION', 2, { min: 1 }))
)
const NEW_BRAIN_MUTATION_STDDEV = readNumberEnv('NEW_BRAIN_MUTATION_STDDEV', 0.02, { min: 0 })
const NEW_BRAIN_MUTATION_JITTER = readNumberEnv('NEW_BRAIN_MUTATION_JITTER', 0.01, { min: 0 })
const NEW_BRAIN_MUTATION_MAX = Math.max(
  NEW_BRAIN_MUTATION_STDDEV,
  readNumberEnv('NEW_BRAIN_MUTATION_MAX', 0.08, { min: 0 })
)
const MIN_CLEAN_BRAIN_SOURCES = Math.max(
  0,
  Math.floor(readNumberEnv('MIN_CLEAN_BRAIN_SOURCES', 5, { min: 0 }))
)
const MUTATION_REWARD_FACTOR = readNumberEnv('REWARD_MUTATION_FACTOR', 0.002, { min: 0 })
const OBS_VALUE_CLAMP = readNumberEnv('OBS_VALUE_CLAMP', 1000, { min: 1 })
const MAX_REWARD_MAGNITUDE = readNumberEnv('MAX_REWARD_MAGNITUDE', 50, { min: 1 })
const ACHIEVEMENT_REWARD_BONUS = Math.max(
  5,
  readNumberEnv('ACHIEVEMENT_REWARD_BONUS', 25, { min: 5 })
)
const GENERATION_SURVIVOR_COUNT = Math.max(
  MIN_BOTS,
  Math.floor(readNumberEnv('GENERATION_SURVIVOR_COUNT', 3, { min: 1 }))
)

const brainWorkerPoolStatus = warmBrainWorkerPool()
if (brainWorkerPoolStatus.enabled) {
  console.log(
    `[Brain] Remote worker pool enabled with ${brainWorkerPoolStatus.size} thread${
      brainWorkerPoolStatus.size === 1 ? '' : 's'
    }.`
  )
} else {
  console.log('[Brain] Remote worker pool disabled; remote calls will run inline.')
}

function monotonicNow() {
  if (typeof performance?.now === 'function') {
    return performance.now()
  }
  return Date.now()
}

function createRollingStats(windowMs = HEALTH_METRIC_WINDOW_MS) {
  const entries = []

  function prune(now = Date.now()) {
    const cutoff = now - windowMs
    while (entries.length && entries[0].time < cutoff) {
      entries.shift()
    }
  }

  return {
    windowMs,
    add(value) {
      if (!Number.isFinite(value)) return
      const now = Date.now()
      prune(now)
      entries.push({ time: now, value })
    },
    summary() {
      prune(Date.now())
      if (!entries.length) {
        return { count: 0, sum: 0, avg: 0, min: 0, max: 0 }
      }
      let sum = 0
      let min = Infinity
      let max = -Infinity
      for (const entry of entries) {
        const val = entry.value
        sum += val
        if (val < min) min = val
        if (val > max) max = val
      }
      return {
        count: entries.length,
        sum,
        avg: sum / entries.length,
        min,
        max
      }
    }
  }
}

const healthMetrics = {
  tickDuration: createRollingStats(),
  remoteFailures: createRollingStats(),
  remoteRecoveries: createRollingStats(),
  sanitization: {
    actionObservation: createRollingStats(),
    actionRemote: createRollingStats(),
    trainObservation: createRollingStats(),
    trainNextObservation: createRollingStats(),
    trainRemote: createRollingStats()
  },
  droppedGradients: createRollingStats(),
  clippedGradients: createRollingStats(),
  gradientNorm: createRollingStats()
}

let nextHealthSummaryAt = Date.now() + HEALTH_SUMMARY_INTERVAL_MS

function extractSanitizationCount(node, seen = new Set()) {
  if (!node || typeof node !== 'object' || seen.has(node)) {
    return 0
  }
  seen.add(node)

  let total = 0
  for (const [key, value] of Object.entries(node)) {
    if (typeof value === 'number' && Number.isFinite(value)) {
      if (
        key.includes('replace') ||
        key.includes('clip') ||
        key.includes('drop') ||
        key.includes('nan') ||
        key.includes('inf') ||
        key.includes('adjust') ||
        key.includes('pad') ||
        key.includes('fill')
      ) {
        total += Math.abs(value)
      }
    } else if (typeof value === 'boolean') {
      if (key.includes('adjust') || key.includes('replace') || key.includes('clip')) {
        total += value ? 1 : 0
      }
    } else if (Array.isArray(value)) {
      total += value.length
    } else if (value && typeof value === 'object') {
      total += extractSanitizationCount(value, seen)
    }
  }

  return total
}

function recordSanitizationMetric(metric, summary) {
  if (!metric || !summary) return
  const total = extractSanitizationCount(summary)
  if (total > 0) {
    metric.add(total)
  }
}

function recordActionSanitization(sanitization) {
  if (!sanitization) return
  recordSanitizationMetric(healthMetrics.sanitization.actionObservation, sanitization.observation)
  recordSanitizationMetric(healthMetrics.sanitization.actionRemote, sanitization.remote)
}

function recordTrainingSanitization(sanitization) {
  if (!sanitization) return
  recordSanitizationMetric(healthMetrics.sanitization.trainObservation, sanitization.observation)
  recordSanitizationMetric(healthMetrics.sanitization.trainNextObservation, sanitization.nextObservation)
  recordSanitizationMetric(healthMetrics.sanitization.trainRemote, sanitization.remote)
}

function recordDroppedGradients(count) {
  if (!Number.isFinite(count) || count <= 0) return
  healthMetrics.droppedGradients.add(count)
}

function recordClippedGradients(count) {
  if (!Number.isFinite(count) || count <= 0) return
  healthMetrics.clippedGradients.add(count)
}

function recordGradientNorm(norm) {
  if (!Number.isFinite(norm) || norm <= 0) return
  healthMetrics.gradientNorm.add(norm)
}

function recordTickDuration(durationMs) {
  if (!Number.isFinite(durationMs) || durationMs < 0) return
  healthMetrics.tickDuration.add(durationMs)
}

function recordRemoteOfflineEvent() {
  healthMetrics.remoteFailures.add(1)
}

function recordRemoteRecoveryEvent() {
  healthMetrics.remoteRecoveries.add(1)
}

function formatAggregate(summary) {
  if (!summary || !summary.count) return '0'
  const total = summary.sum
  const formattedTotal = Number.isInteger(total) ? total : Number(total.toFixed(2))
  return `${formattedTotal} (events ${summary.count})`
}

function maybeLogHealthSummary() {
  const now = Date.now()
  if (now < nextHealthSummaryAt) return
  nextHealthSummaryAt = now + HEALTH_SUMMARY_INTERVAL_MS

  const tickSummary = healthMetrics.tickDuration.summary()
  const offlineSummary = healthMetrics.remoteFailures.summary()
  const recoverySummary = healthMetrics.remoteRecoveries.summary()
  const actionObsSummary = healthMetrics.sanitization.actionObservation.summary()
  const actionRemoteSummary = healthMetrics.sanitization.actionRemote.summary()
  const trainObsSummary = healthMetrics.sanitization.trainObservation.summary()
  const trainNextSummary = healthMetrics.sanitization.trainNextObservation.summary()
  const trainRemoteSummary = healthMetrics.sanitization.trainRemote.summary()
  const droppedSummary = healthMetrics.droppedGradients.summary()
  const clippedSummary = healthMetrics.clippedGradients.summary()
  const gradientNormSummary = healthMetrics.gradientNorm.summary()

  const windowSeconds = Math.round(HEALTH_METRIC_WINDOW_MS / 1000)
  const tickAvg = tickSummary.count ? tickSummary.avg.toFixed(1) : 'n/a'
  const tickMax = tickSummary.count ? tickSummary.max.toFixed(1) : 'n/a'
  const gradNormAvg = gradientNormSummary.count ? gradientNormSummary.avg.toFixed(2) : 'n/a'
  const gradNormMax = gradientNormSummary.count ? gradientNormSummary.max.toFixed(2) : 'n/a'

  console.log(
    `[Health] Last ${windowSeconds}s | Tick avg ${tickAvg}ms (max ${tickMax}ms, n=${tickSummary.count}) | ` +
      `Remote offline ${formatAggregate(offlineSummary)} | Remote recoveries ${formatAggregate(recoverySummary)} | ` +
      `Act sanitize obs=${formatAggregate(actionObsSummary)}, remote=${formatAggregate(actionRemoteSummary)} | ` +
      `Train sanitize obs=${formatAggregate(trainObsSummary)}, next=${formatAggregate(trainNextSummary)}, remote=${formatAggregate(trainRemoteSummary)} | ` +
      `Dropped grads ${formatAggregate(droppedSummary)} | Clipped grads ${formatAggregate(clippedSummary)} | ` +
      `Grad norm avg ${gradNormAvg} (max ${gradNormMax})`
  )
}

function sanitizeScalar(value, clamp = OBS_VALUE_CLAMP, fallback = 0) {
  if (!Number.isFinite(value)) {
    return fallback
  }
  if (clamp > 0) {
    if (value > clamp) return clamp
    if (value < -clamp) return -clamp
  }
  return value
}

function sanitizeVector(vector, clamp = OBS_VALUE_CLAMP) {
  if (!vector || typeof vector.length !== 'number') {
    return vector
  }
  for (let i = 0; i < vector.length; i++) {
    vector[i] = sanitizeScalar(vector[i], clamp, 0)
  }
  return vector
}

function vectorHasFiniteValues(vector) {
  if (!vector || typeof vector.length !== 'number') return false
  for (let i = 0; i < vector.length; i++) {
    if (!Number.isFinite(vector[i])) {
      return false
    }
  }
  return true
}

function clampReward(value) {
  if (!Number.isFinite(value)) {
    return 0
  }
  const limit = MAX_REWARD_MAGNITUDE
  if (value > limit) return limit
  if (value < -limit) return -limit
  return value
}

function deriveMutationStddev(preferred, { jitter = true } = {}) {
  let base = Number.isFinite(preferred) && preferred >= 0 ? preferred : NEW_BRAIN_MUTATION_STDDEV
  if (!Number.isFinite(base) || base < 0) {
    base = NEW_BRAIN_MUTATION_STDDEV
  }
  let result = base
  if (jitter && NEW_BRAIN_MUTATION_JITTER > 0) {
    result += (Math.random() - 0.5) * NEW_BRAIN_MUTATION_JITTER
  }
  if (!Number.isFinite(result) || result <= 0) {
    return null
  }
  if (NEW_BRAIN_MUTATION_MAX > 0) {
    result = Math.min(NEW_BRAIN_MUTATION_MAX, result)
  }
  return result
}

function enqueuePendingMutation(context, stddev) {
  if (!context) return
  if (!Number.isFinite(stddev) || stddev <= 0) {
    return
  }
  const normalized = Math.min(NEW_BRAIN_MUTATION_MAX, Math.max(0, stddev))
  if (normalized <= 0) return
  if (!Array.isArray(context.pendingMutations)) {
    context.pendingMutations = []
  }
  context.pendingMutations.push(normalized)
}

const pipelineAsync = promisify(pipeline)

class RotatingCompressedLogger {
  constructor(directory, {
    maxBytes = 512 * 1024,
    maxAgeMs = 5 * 60 * 1000
  } = {}) {
    this.directory = directory
    this.maxBytes = maxBytes
    this.maxAgeMs = maxAgeMs
    this.currentStream = null
    this.currentGzip = null
    this.currentPath = null
    this.currentSize = 0
    this.openedAt = 0
    fs.mkdirSync(this.directory, { recursive: true })
  }

  _shouldRotate() {
    if (!this.currentStream) return true
    if (this.currentSize >= this.maxBytes) return true
    if (Date.now() - this.openedAt >= this.maxAgeMs) return true
    return false
  }

  async _rotate() {
    if (this.currentGzip) {
      await new Promise(resolve => {
        this.currentGzip.once('finish', resolve)
        this.currentGzip.end()
      })
      this.currentGzip = null
      this.currentStream = null
      this.currentPath = null
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
    const filename = `brain-${timestamp}.log.gz`
    const filePath = path.join(this.directory, filename)
    const fileStream = fs.createWriteStream(filePath, { flags: 'w' })
    const gzip = zlib.createGzip({ level: zlib.constants.Z_BEST_SPEED })
    pipelineAsync(gzip, fileStream).catch(err => {
      originalConsole.error('[Logger] Failed to pipeline log stream:', err)
    })

    this.currentStream = fileStream
    this.currentGzip = gzip
    this.currentPath = filePath
    this.currentSize = 0
    this.openedAt = Date.now()
  }

  async write(level, message) {
    try {
      if (this._shouldRotate()) {
        await this._rotate()
      }

      if (!this.currentGzip) {
        await this._rotate()
      }

      const line = `[${new Date().toISOString()}] [${level}] ${message}\n`
      const buffer = Buffer.from(line, 'utf8')
      this.currentSize += buffer.length
      this.currentGzip.write(buffer)
    } catch (err) {
      originalConsole.error('[Logger] Failed to write log entry:', err)
    }
  }
}

const originalConsole = {
  log: console.log.bind(console),
  warn: console.warn.bind(console),
  error: console.error.bind(console)
}

const LOG_DIR = path.join(process.cwd(), 'logs')
const logger = new RotatingCompressedLogger(LOG_DIR)

async function mirrorToLog(level, args) {
  try {
    const message = args
      .map(value => (typeof value === 'string' ? value : JSON.stringify(value)))
      .join(' ')
    await logger.write(level, message)
  } catch (err) {
    originalConsole.error('[Logger] Mirror failed:', err)
  }
}

console.log = (...args) => {
  originalConsole.log(...args)
  mirrorToLog('INFO', args)
}
console.warn = (...args) => {
  originalConsole.warn(...args)
  mirrorToLog('WARN', args)
}
console.error = (...args) => {
  originalConsole.error(...args)
  mirrorToLog('ERROR', args)
}

function isRemoteBrainUnavailableError(err) {
  return err instanceof RemoteBrainUnavailableError || err?.code === 'REMOTE_BRAIN_UNAVAILABLE'
}

function describeRemoteRetry(status) {
  if (!status) return 'retry pending'
  if (status.retryAt) {
    return `retry after ${new Date(status.retryAt).toISOString()}`
  }
  if (status.retryDelay) {
    return `retry delay ${Math.round(status.retryDelay)}ms`
  }
  return 'retry pending'
}

function ensureRemoteBrainTracker(context) {
  if (!context.remoteBrain) {
    context.remoteBrain = {
      offlineNotified: false,
      lastMessage: null,
      nextLogAt: 0
    }
  }
  return context.remoteBrain
}

function noteRemoteBrainOffline(context, status, source) {
  const tracker = ensureRemoteBrainTracker(context)
  const now = Date.now()
  const message = source?.message ?? status?.lastError ?? 'Remote brain unavailable'
  if (!tracker.offlineNotified || tracker.lastMessage !== message || now >= tracker.nextLogAt) {
    const retryNote = describeRemoteRetry(status)
    console.warn(`[${label(context)}] Remote brain unavailable: ${message} (${retryNote}).`)
    if (!tracker.offlineNotified) {
      recordRemoteOfflineEvent()
    }
    tracker.offlineNotified = true
    tracker.lastMessage = message
    tracker.nextLogAt = now + 5000
  }
  context.waitingForBrain = true
}

function noteRemoteBrainOnline(context) {
  const tracker = ensureRemoteBrainTracker(context)
  if (tracker.offlineNotified) {
    console.log(`[${label(context)}] Remote brain connection restored. Resuming ticks.`)
    recordRemoteRecoveryEvent()
  }
  tracker.offlineNotified = false
  tracker.lastMessage = null
  tracker.nextLogAt = 0
  context.waitingForBrain = false
}

function ensureNonFiniteTracker(context) {
  if (!context) return null
  if (!context.nonFiniteTracker) {
    context.nonFiniteTracker = {
      act: 0,
      train: 0,
      total: 0,
      escalations: 0,
      lastReset: Date.now(),
      lastReason: null,
      lastEscalationAt: 0
    }
  }
  return context.nonFiniteTracker
}

function resetNonFiniteTracker(context) {
  const tracker = ensureNonFiniteTracker(context)
  if (!tracker) return
  tracker.act = 0
  tracker.train = 0
  tracker.total = 0
  tracker.escalations = 0
  tracker.lastReason = null
  tracker.lastEscalationAt = 0
  tracker.lastReset = Date.now()
}

function registerNonFiniteStrike(context, reason) {
  const tracker = ensureNonFiniteTracker(context)
  if (!tracker) return
  const now = Date.now()
  if (now - tracker.lastReset > NON_FINITE_STRIKE_WINDOW_MS) {
    tracker.act = 0
    tracker.train = 0
    tracker.total = 0
    tracker.escalations = 0
    tracker.lastReset = now
  }
  if (reason === 'train-non-finite') {
    tracker.train = (tracker.train ?? 0) + 1
  } else {
    tracker.act = (tracker.act ?? 0) + 1
  }
  tracker.total = (tracker.total ?? 0) + 1
  tracker.lastReason = reason

  const strikesForReason = reason === 'train-non-finite' ? tracker.train : tracker.act
  const needsEscalation =
    strikesForReason >= NON_FINITE_REASON_THRESHOLD || tracker.total >= NON_FINITE_TOTAL_THRESHOLD
  if (!needsEscalation) {
    return
  }

  const pending = context.pendingWeightRecovery
  if (!pending) {
    return
  }

  tracker.escalations = (tracker.escalations ?? 0) + 1
  tracker.lastEscalationAt = now

  const baseStddev = deriveMutationStddev(null, { jitter: false }) ?? NEW_BRAIN_MUTATION_STDDEV
  const escalationFactor = Math.max(1, 1 + NON_FINITE_MUTATION_MULTIPLIER * tracker.escalations)
  const escalatedStddev = Math.min(
    NEW_BRAIN_MUTATION_MAX,
    Math.max(NON_FINITE_MUTATION_MIN, baseStddev * escalationFactor)
  )
  if (Number.isFinite(escalatedStddev) && escalatedStddev > 0) {
    if (!Array.isArray(pending.extraMutations)) {
      pending.extraMutations = []
    }
    pending.extraMutations.push(escalatedStddev)
    console.warn(
      `[${label(context)}] Escalating ${reason} recovery after ${tracker.total} strike(s); adding mutation ${escalatedStddev.toFixed(
        3
      )}.`
    )
  }

  context.epsilonBoost = Math.max(context.epsilonBoost ?? 0, NON_FINITE_EPSILON_BOOST)

  if (tracker.escalations >= NON_FINITE_MAX_ESCALATIONS) {
    pending.forceReinitialize = true
  }
}

function markBaselineWeightsHealthy(reason = 'unknown') {
  baselineWeightsSuspect = false
}

function markContextWeightsHealthy(context, reason = 'unknown') {
  if (!context) return
  context.weightsSuspect = false
  context.pendingWeightRecovery = null
  context.weightRecoveryInFlight = null
  context.lastWeightIssue = null
  context.lastWeightRecovery = Date.now()
  context.lastWeightRecoveryReason = reason
  context.weightSkipNotified = false
  resetNonFiniteTracker(context)
  if (context.lineage) {
    const stats = ensureLineageRecord(context.lineage)
    stats.instability = Math.max(0, (stats.instability ?? 0) * 0.5)
  }
}

function scheduleWeightRecovery(context, reason = 'unknown', details = {}) {
  if (!context) return
  const firstDetection = !context.weightsSuspect
  context.weightsSuspect = true
  context.lastWeightIssue = {
    reason,
    details,
    detectedAt: Date.now()
  }
  context.weightSkipNotified = false
  if (
    !context.pendingWeightRecovery ||
    context.pendingWeightRecovery.reason !== reason
  ) {
    context.pendingWeightRecovery = {
      reason,
      details,
      attempts: 0,
      scheduledAt: Date.now(),
      extraMutations: Array.isArray(context.pendingWeightRecovery?.extraMutations)
        ? [...context.pendingWeightRecovery.extraMutations]
        : [],
      forceReinitialize: Boolean(context.pendingWeightRecovery?.forceReinitialize)
    }
  }
  if (reason === 'act-non-finite' || reason === 'train-non-finite') {
    registerNonFiniteStrike(context, reason)
    if (context.lineage) {
      const stats = ensureLineageRecord(context.lineage)
      stats.instability = (stats.instability ?? 0) + 1
    }
  }
  if (firstDetection) {
    console.warn(
      `[${label(context)}] Detected suspect brain weights (${reason}); scheduling recovery.`
    )
  }
}

function gatherCleanBrainSources({ exclude = [] } = {}) {
  const excludeSet = new Set(
    Array.isArray(exclude) ? exclude.filter(Boolean) : [exclude].filter(Boolean)
  )
  const sources = []
  for (const ctx of contexts) {
    if (!ctx?.brain?.id) continue
    if (excludeSet.has(ctx)) continue
    if (ctx.weightsSuspect) continue
    sources.push(ctx.brain)
  }
  if (baselineBrain && baselineBrain.id && !baselineWeightsSuspect) {
    sources.push(baselineBrain)
  }
  return sources
}

async function reinitializeContextBrain(context, reason = 'reinitialize', details = {}) {
  if (!context) {
    return { success: false, sourceCount: 0, mode: 'none' }
  }

  try {
    await ensureBaselineReady()
  } catch (err) {
    if (isRemoteBrainUnavailableError(err)) {
      throw err
    }
    console.error(`[${label(context)}] Failed to prepare baseline during ${reason}:`, err)
    return { success: false, sourceCount: 0, mode: 'reinitialize-failed' }
  }

  let brain
  try {
    brain = await createBrain(OBS_SIZE, ACTIONS.length)
  } catch (err) {
    if (isRemoteBrainUnavailableError(err)) {
      throw err
    }
    console.error(`[${label(context)}] Failed to allocate replacement brain during ${reason}:`, err)
    return { success: false, sourceCount: 0, mode: 'reinitialize-failed' }
  }

  brain.owner = label(context)

  if (baselineBrain && baselineBrain.id && brain.id && brain.id !== baselineBrain.id) {
    try {
      await copyWeights(brain, baselineBrain)
    } catch (err) {
      if (isRemoteBrainUnavailableError(err)) {
        throw err
      }
      console.error(
        `[${label(context)}] Failed to seed replacement brain from baseline during ${reason}:`,
        err
      )
    }
  }

  if (!Array.isArray(context.pendingMutations)) {
    context.pendingMutations = []
  }

  const mutationQueue = [...context.pendingMutations]
  context.pendingMutations = []

  for (let i = 0; i < mutationQueue.length; i++) {
    const stddev = Math.min(
      NEW_BRAIN_MUTATION_MAX,
      Math.max(0, mutationQueue[i] ?? 0)
    )
    if (!Number.isFinite(stddev) || stddev <= 0) {
      continue
    }
    try {
      await mutateWeights(brain, stddev)
    } catch (err) {
      if (isRemoteBrainUnavailableError(err)) {
        const remaining = mutationQueue.slice(i)
        context.pendingMutations = remaining.concat(context.pendingMutations)
        throw err
      }
      console.error(
        `[${label(context)}] Failed to apply pending mutation (${stddev}) during ${reason}:`,
        err
      )
    }
  }

  context.brain = brain
  markContextWeightsHealthy(context, reason)
  console.warn(
    `[${label(context)}] Reinitialized brain from baseline after ${reason}.`,
    details
  )
  return { success: true, sourceCount: 0, mode: 'reinitialize' }
}

async function rebuildContextWeights(context, reason = 'unknown', details = {}) {
  if (!context?.brain?.id) {
    return { success: false, sourceCount: 0, mode: 'none' }
  }

  const exclude = new Set([context])
  let sources = gatherCleanBrainSources({ exclude })

  if (sources.length < MIN_CLEAN_BRAIN_SOURCES) {
    console.warn(
      `[${label(context)}] Only ${sources.length} clean brain source${
        sources.length === 1 ? '' : 's'
      } available; reinitializing for ${reason}.`
    )
    try {
      return await reinitializeContextBrain(context, reason, details)
    } catch (err) {
      if (isRemoteBrainUnavailableError(err)) {
        throw err
      }
      console.error(
        `[${label(context)}] Brain reinitialization failed during ${reason}:`,
        err
      )
      return { success: false, sourceCount: sources.length, mode: 'reinitialize-failed' }
    }
  }

  if (!sources.length) {
    try {
      await ensureBaselineReady()
    } catch (err) {
      if (isRemoteBrainUnavailableError(err)) {
        throw err
      }
      console.error('[Baseline] Baseline not ready for weight recovery:', err)
    }
    if (baselineBrain && baselineBrain.id && !baselineWeightsSuspect) {
      sources = [baselineBrain]
    }
  }

  if (!sources.length) {
    const fallbackStd =
      deriveMutationStddev(null, { jitter: true }) ?? NEW_BRAIN_MUTATION_STDDEV
    if (!Number.isFinite(fallbackStd) || fallbackStd <= 0) {
      return { success: false, sourceCount: 0, mode: 'none' }
    }
    try {
      const stddev = Math.min(
        NEW_BRAIN_MUTATION_MAX,
        Math.max(0.01, fallbackStd)
      )
      console.warn(
        `[${label(context)}] No clean brain sources available; applying fallback mutation for ${reason}.`
      )
      await mutateWeights(context.brain, stddev)
      return { success: true, sourceCount: 0, mode: 'mutate' }
    } catch (err) {
      if (isRemoteBrainUnavailableError(err)) {
        throw err
      }
      console.error(
        `[${label(context)}] Fallback mutation during weight recovery failed:`,
        err
      )
      return { success: false, sourceCount: 0, mode: 'mutate-failed' }
    }
  }

  try {
    if (sources.length === 1) {
      await copyWeights(context.brain, sources[0])
      return { success: true, sourceCount: 1, mode: 'copy' }
    }
    await averageWeights(context.brain, sources)
    return { success: true, sourceCount: sources.length, mode: 'average' }
  } catch (err) {
    if (isRemoteBrainUnavailableError(err)) {
      throw err
    }
    console.error(
      `[${label(context)}] Failed to rebuild weights from ${sources.length} clean source(s):`,
      err
    )
    return { success: false, sourceCount: sources.length, mode: 'average-failed' }
  }
}

async function attemptWeightRecovery(context, trigger = 'tick') {
  if (!context?.pendingWeightRecovery) {
    return false
  }
  if (context.weightRecoveryInFlight) {
    return context.weightRecoveryInFlight
  }

  if (!isRemoteBrainConnected()) {
    const status = getRemoteBrainStatus()
    console.warn(
      `[${label(context)}] Weight recovery deferred (${context.pendingWeightRecovery.reason}, trigger=${trigger}) while remote brain unavailable (${describeRemoteRetry(status)}).`
    )
    return false
  }

  const pending = context.pendingWeightRecovery
  context.weightRecoveryInFlight = (async () => {
    let outcome
    try {
      if (pending.forceReinitialize) {
        outcome = await reinitializeContextBrain(context, `${pending.reason}-force`, pending.details)
      } else {
        outcome = await rebuildContextWeights(context, pending.reason, pending.details)
      }
    } catch (err) {
      if (isRemoteBrainUnavailableError(err)) {
        const status = getRemoteBrainStatus()
        console.warn(
          `[${label(context)}] Weight recovery postponed (${pending.reason}, trigger=${trigger}) while remote brain unavailable (${describeRemoteRetry(status)}).`
        )
      } else {
        console.error(
          `[${label(context)}] Weight recovery failed during ${pending.reason}:`,
          err
        )
      }
      context.pendingWeightRecovery = pending
      return false
    }

    if (outcome?.success) {
      markContextWeightsHealthy(context, pending.reason)
      if (Array.isArray(pending.extraMutations) && pending.extraMutations.length) {
        const extras = pending.extraMutations.splice(0)
        for (const stddev of extras) {
          if (!Number.isFinite(stddev) || stddev <= 0) continue
          try {
            await mutateWeights(context.brain, Math.min(NEW_BRAIN_MUTATION_MAX, Math.max(0, stddev)))
          } catch (err) {
            if (isRemoteBrainUnavailableError(err)) {
              pending.extraMutations.unshift(stddev)
              throw err
            }
            console.error(
              `[${label(context)}] Failed to apply escalated mutation (${stddev}) after ${pending.reason}:`,
              err
            )
          }
        }
      }
      const sourceNote =
        outcome.mode === 'copy' || outcome.mode === 'average'
          ? ` from ${outcome.sourceCount} clean source${
              outcome.sourceCount === 1 ? '' : 's'
            }`
          : ''
      const modeLabel =
        outcome.mode === 'mutate'
          ? 'via fallback mutation'
          : `via ${outcome.mode}`
      console.warn(
        `[${label(context)}] Rebuilt brain weights ${modeLabel}${sourceNote} after ${pending.reason} (trigger=${trigger}).`
      )
      return true
    }

    pending.attempts = (pending.attempts ?? 0) + 1
    pending.lastAttemptAt = Date.now()
    context.pendingWeightRecovery = pending
    console.warn(
      `[${label(context)}] Weight recovery attempt #${pending.attempts} did not succeed (${pending.reason}, trigger=${trigger}).`
    )
    return false
  })()

  try {
    return await context.weightRecoveryInFlight
  } finally {
    context.weightRecoveryInFlight = null
  }
}

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
  'mine_forward',
  'strafe_mine_left',
  'strafe_mine_right',
  'attack',
  'build',
  'build_above',
  'build_forward',
  'use_item',
  'signal_resource',
  'signal_danger',
  'signal_assist',
  'signal_gather',
  'signal_status',
  ...Object.keys(CRAFTING_ACTIONS)
]

const TICK_RATE = 50
const EPSILON_START = 0.25
const EPSILON_MIN = 0.05
const EPSILON_DECAY = 0.999
const EPSILON_STAGNATION_BOOST = 0.4
const EPSILON_BOOST_DECAY = 0.995
const SAVE_INTERVAL_TICKS = 40
const SAVE_INTERVAL_MS = 60 * 1000
const CHECKPOINT_DIR = DEFAULT_BRAIN_DIR
const EMOTION_VECTOR_SIZE = 3
const COMMUNICATION_TYPES = ['resource', 'danger', 'assist', 'gather', 'status']
const COMMUNICATION_TYPE_COUNT = COMMUNICATION_TYPES.length
const COMMUNICATION_TYPE_INDEX = new Map(
  COMMUNICATION_TYPES.map((type, index) => [type, index])
)
const COMMUNICATION_RANGE = readNumberEnv('BOT_COMMUNICATION_RANGE', 96, { min: 4, max: 512 })
const COMMUNICATION_DECAY_MS = readNumberEnv('BOT_COMMUNICATION_DECAY_MS', 15000, {
  min: 250,
  max: 60000
})
const COMMUNICATION_COOLDOWN_MS = readNumberEnv('BOT_COMMUNICATION_COOLDOWN_MS', 1200, {
  min: 100,
  max: 60000
})
const COMMUNICATION_MESSAGE_LIMIT = Math.max(
  COMMUNICATION_TYPE_COUNT * 8,
  Math.floor(readNumberEnv('BOT_COMMUNICATION_MESSAGE_LIMIT', 320, { min: 32, max: 2000 }))
)
const COMMUNICATION_PRUNE_INTERVAL_MS = readNumberEnv('BOT_COMMUNICATION_PRUNE_INTERVAL', 500, {
  min: 50,
  max: 5000
})
const COMMUNICATION_INTENSITY_CLAMP = readNumberEnv('BOT_COMMUNICATION_INTENSITY_CLAMP', 3, {
  min: 0.5,
  max: 10
})
const COMMUNICATION_TIME_NORMALIZER = readNumberEnv('BOT_COMMUNICATION_TIME_NORMALIZER', 12000, {
  min: 250,
  max: 120000
})
const COMMUNICATION_DIRECTION_NORMALIZER = readNumberEnv(
  'BOT_COMMUNICATION_DIRECTION_NORMALIZER',
  COMMUNICATION_RANGE,
  { min: 1, max: 512 }
)
const COMMUNICATION_REWARD = readNumberEnv('BOT_COMMUNICATION_REWARD', 0.15, {
  min: 0,
  max: 1
})
const COMMUNICATION_COOLDOWN_PENALTY = readNumberEnv('BOT_COMMUNICATION_COOLDOWN_PENALTY', 0.05, {
  min: 0,
  max: 1
})

const BASE_OBS_FEATURES = 70
const MEMORY_REWARD_WINDOW = Math.max(
  4,
  Math.floor(readNumberEnv('BOT_MEMORY_REWARD_WINDOW', 16, { min: 4, max: 64 }))
)
const MEMORY_ACTION_WINDOW = Math.max(
  4,
  Math.floor(readNumberEnv('BOT_MEMORY_ACTION_WINDOW', 16, { min: 4, max: 64 }))
)
const MEMORY_AGGREGATE_COUNT = 13
const MEMORY_OBS_SIZE = MEMORY_REWARD_WINDOW + MEMORY_ACTION_WINDOW + MEMORY_AGGREGATE_COUNT
const MEMORY_OBS_START = BASE_OBS_FEATURES
const OBS_SIZE = BASE_OBS_FEATURES + MEMORY_OBS_SIZE
const NOVELTY_HASH_PRECISION = 2
const NOVELTY_TARGET = 500
const CONTEXT_TICK_NORMALIZER = readNumberEnv('BOT_CONTEXT_TICK_NORMALIZER', 2400, { min: 240, max: 48000 })
const CONTEXT_TRAINING_NORMALIZER = readNumberEnv('BOT_CONTEXT_TRAINING_NORMALIZER', 600, { min: 60, max: 40000 })
const CONTEXT_REWARD_NORMALIZER = readNumberEnv('BOT_CONTEXT_REWARD_NORMALIZER', 300, { min: 10, max: 5000 })
const NOVELTY_COUNT_NORMALIZER = readNumberEnv('BOT_NOVELTY_COUNT_NORMALIZER', NOVELTY_TARGET, { min: 50, max: 5000 })
const MUTATION_QUEUE_CLAMP = readNumberEnv('BOT_MUTATION_QUEUE_CLAMP', 12, { min: 1, max: 128 })
const EPSILON_BOOST_NORMALIZER = readNumberEnv('BOT_EPSILON_BOOST_NORMALIZER', 1, { min: 0.05, max: 10 })
const EPSILON_NORMALIZER = readNumberEnv('BOT_EPSILON_NORMALIZER', 1, { min: 0.1, max: 2 })
const FERAL_FURY_NORMALIZER = readNumberEnv('BOT_FERAL_FURY_NORMALIZER', 5, { min: 0.5, max: 20 })
const COOPERATION_CLAMP = readNumberEnv('BOT_COOPERATION_CLAMP', 1.5, { min: 0.1, max: 10 })
const STREAK_NORMALIZER = readNumberEnv('BOT_STREAK_NORMALIZER', 24, { min: 1, max: 200 })
const REPETITION_NORMALIZER = readNumberEnv('BOT_REPETITION_NORMALIZER', 24, { min: 1, max: 200 })
const MORALE_FRUSTRATION_CLAMP = readNumberEnv('BOT_MORALE_FRUSTRATION_CLAMP', 2, { min: 0.1, max: 10 })
const MORALE_SHARPNESS_CLAMP = readNumberEnv('BOT_MORALE_SHARPNESS_CLAMP', 2, { min: 0.1, max: 10 })
const STAGNATION_WINDOW = 40
const STAGNATION_VARIANCE_THRESHOLD = 0.0025
const STAGNATION_MUTATION_THRESHOLD = 3
const MEMORY_SHORT_DECAY = readNumberEnv('BOT_MEMORY_SHORT_DECAY', 0.35, { min: 0.05, max: 0.95 })
const MEMORY_LONG_DECAY = readNumberEnv('BOT_MEMORY_LONG_DECAY', 0.08, { min: 0.01, max: 0.6 })
const MEMORY_VOLATILITY_DECAY = readNumberEnv('BOT_MEMORY_VOLATILITY_DECAY', 0.18, { min: 0.01, max: 0.6 })
const REWARD_MEMORY_CLAMP = readNumberEnv('BOT_MEMORY_REWARD_CLAMP', 3, { min: 0.5, max: 10 })
const ACTION_MEMORY_RATE = readNumberEnv('BOT_ACTION_MEMORY_RATE', 0.25, { min: 0.05, max: 0.95 })
const ACTION_VALUE_CLAMP = readNumberEnv('BOT_ACTION_VALUE_CLAMP', 2, { min: 0.5, max: 8 })
const ACTION_MEMORY_EPSILON = 0.001
const DAMAGE_DEBT_DECAY = readNumberEnv('BOT_DAMAGE_DEBT_DECAY', 0.88, { min: 0.1, max: 0.999 })
const DAMAGE_RECENT_DECAY = readNumberEnv('BOT_DAMAGE_RECENT_DECAY', 0.75, { min: 0.1, max: 0.999 })
const DAMAGE_DEBT_WEIGHT = readNumberEnv('BOT_DAMAGE_DEBT_WEIGHT', 0.6, { min: 0, max: 5 })
const DAMAGE_DEBT_PENALTY = readNumberEnv('BOT_DAMAGE_DEBT_PENALTY', 0.4, { min: 0, max: 5 })
const DAMAGE_MEMORY_CLAMP = readNumberEnv('BOT_DAMAGE_MEMORY_CLAMP', 10, { min: 1, max: 40 })
const DIVERSITY_WINDOW = 60
const MAX_BOTS = Math.max(BOT_COUNT, parseInt(process.env.BOT_MAX ?? '8', 10))
const LOW_REWARD_RETIRE_LIMIT = (() => {
  const raw = Number.parseInt(process.env.BOT_LOW_REWARD_RETIRE ?? '2', 10)
  if (!Number.isFinite(raw) || raw <= 0) return 0
  return raw
})()
const LOW_REWARD_RETIRE_THRESHOLD = (() => {
  const raw = Number.parseFloat(process.env.BOT_LOW_REWARD_THRESHOLD ?? '')
  if (Number.isFinite(raw)) return raw
  return Number.NEGATIVE_INFINITY
})()
const DOWNTREND_GENERATION_WINDOW = (() => {
  const raw = Number.parseInt(process.env.BOT_DOWNTREND_WINDOW ?? '4', 10)
  return Number.isFinite(raw) && raw >= 2 ? Math.max(3, raw) : 4
})()
const DOWNTREND_SLOPE_THRESHOLD = (() => {
  const raw = Number.parseFloat(process.env.BOT_DOWNTREND_SLOPE ?? '0.35')
  return Number.isFinite(raw) ? Math.max(0.05, raw) : 0.35
})()
const DOWNTREND_MARGIN = (() => {
  const raw = Number.parseFloat(process.env.BOT_DOWNTREND_MARGIN ?? '0.3')
  return Number.isFinite(raw) ? Math.max(0, raw) : 0.3
})()
const DOWNTREND_STREAK_LIMIT = (() => {
  const raw = Number.parseInt(process.env.BOT_DOWNTREND_STREAK ?? '2', 10)
  return Number.isFinite(raw) && raw >= 1 ? raw : 2
})()
const DOWNTREND_MIN_AGE_GENERATIONS = (() => {
  const raw = Number.parseInt(process.env.BOT_DOWNTREND_MIN_AGE ?? '3', 10)
  return Number.isFinite(raw) && raw >= 1 ? raw : 3
})()
const DOWNTREND_RETIRE_LIMIT = (() => {
  const raw = Number.parseInt(process.env.BOT_DOWNTREND_RETIRE ?? '2', 10)
  return Number.isFinite(raw) && raw >= 1 ? raw : 2
})()
const MAX_USERNAME_LENGTH = 16
const RESOURCE_TYPES = ['wood', 'stone', 'ore', 'crafted']

// ----------------------------
// GLOBAL STATE
// ----------------------------
const contexts = []
const usedNames = new Set()
const lineageStats = new Map()
const lineageCounters = new Map([[LINEAGE_ROOT_NAME, 0]])
let birthCounter = 0
let globalRunning = true
let baselineBrain = null
let baselineReady = null
let baselineWeightsSuspect = false
let deferredBaselineSaveReason = null
let nextBaselineSaveLogAt = 0
const GLOBAL_RESOURCE_POOL = {
  wood: 0,
  stone: 0,
  ore: 0,
  crafted: 0,
  totalContribution: 0,
  totalWithdrawal: 0,
  diversity: new Set()
}
const COMMUNICATION_BUS = {
  messages: [],
  lastPrune: 0
}
const rewardProfile = {
  novelty: 1,
  cooperation: 1,
  entropy: 1,
  skill: 1,
  resource: 1,
  lineage: 1,
  feral: 1.15,
  morale: 1
}
let nextRewardMutationTick = REWARD_MUTATION_INTERVAL
const crossoverBrains = []
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
let stagnantGenerations = 0
let bestGenerationReward = -Infinity
let fatalRecoveryInProgress = false
let fatalRecoveryAttempts = 0
let lastFatalRecoveryAt = 0

// ----------------------------
// HELPERS
// ----------------------------
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms))

function clampNumber(value, min, max) {
  if (!Number.isFinite(value)) {
    if (Number.isNaN(value)) return 0
    return value > 0 ? max : value < 0 ? min : 0
  }
  return Math.min(max, Math.max(min, value))
}

function normalizeAngle(angle) {
  if (!Number.isFinite(angle)) return 0
  let wrapped = angle % (Math.PI * 2)
  if (wrapped > Math.PI) wrapped -= Math.PI * 2
  if (wrapped < -Math.PI) wrapped += Math.PI * 2
  return wrapped
}

function createDefaultMovementModifiers() {
  return {
    sprint: false,
    sneak: false,
    jump: false,
    keepJump: false
  }
}

function createMovementController(context) {
  const state = {
    context,
    current: { forward: 0, strafe: 0 },
    target: { forward: 0, strafe: 0 },
    modifiers: createDefaultMovementModifiers(),
    lastUpdate: monotonicNow(),
    jumpReleaseAt: 0,
    token: null,
    activeControls: {
      forward: false,
      back: false,
      left: false,
      right: false,
      sprint: false,
      sneak: false,
      jump: false
    }
  }

  function applyControl(name, value) {
    const bot = state.context?.bot
    if (!bot || typeof bot.setControlState !== 'function') return
    if (state.activeControls[name] === value) return
    state.activeControls[name] = value
    try {
      bot.setControlState(name, value)
    } catch (err) {
      console.warn(`[${label(state.context)}] Failed to apply control ${name}:`, err?.message ?? err)
    }
  }

  function setTarget(vector = {}, options = {}, token = null) {
    state.target.forward = clampNumber(Number(vector.forward ?? 0), -1, 1)
    state.target.strafe = clampNumber(Number(vector.strafe ?? 0), -1, 1)
    if (token != null) {
      state.token = token
    }
    const nextModifiers = createDefaultMovementModifiers()
    if (options && typeof options === 'object') {
      if (options.sprint) nextModifiers.sprint = true
      if (options.sneak) nextModifiers.sneak = true
      if (options.jump) nextModifiers.jump = true
      if (options.keepJump) nextModifiers.keepJump = true
    }
    state.modifiers = nextModifiers
    if (state.modifiers.jump) {
      const holdMs = clampNumber(Number(options?.jumpHoldMs ?? MOVEMENT_JUMP_MAX_MS), 0, MOVEMENT_JUMP_MAX_MS)
      state.jumpReleaseAt = monotonicNow() + holdMs
    } else if (!state.modifiers.keepJump) {
      state.jumpReleaseAt = 0
    }
  }

  function clearTarget(token = null, { immediate = false } = {}) {
    if (token != null && state.token != null && token !== state.token) {
      return false
    }
    state.target.forward = 0
    state.target.strafe = 0
    state.modifiers = createDefaultMovementModifiers()
    state.jumpReleaseAt = 0
    if (token == null || state.token === token) {
      state.token = null
    }
    if (immediate) {
      state.current.forward = 0
      state.current.strafe = 0
    }
    return true
  }

  function update(now = monotonicNow()) {
    const dt = Math.max(0, now - state.lastUpdate)
    state.lastUpdate = now
    const ramp = MOVEMENT_SMOOTH_RAMP_MS <= 0 ? 1 : Math.min(1, dt / MOVEMENT_SMOOTH_RAMP_MS)
    state.current.forward += (state.target.forward - state.current.forward) * ramp
    state.current.strafe += (state.target.strafe - state.current.strafe) * ramp

    const threshold = MOVEMENT_SMOOTH_THRESHOLD
    const forwardValue = state.current.forward
    const strafeValue = state.current.strafe

    const forwardActive = forwardValue > threshold
    const backActive = forwardValue < -threshold
    const rightActive = strafeValue > threshold
    const leftActive = strafeValue < -threshold

    applyControl('forward', forwardActive)
    applyControl('back', backActive)
    applyControl('right', rightActive)
    applyControl('left', leftActive)

    const shouldSprint = Boolean(state.modifiers.sprint) && !Boolean(state.modifiers.sneak)
    const shouldSneak = Boolean(state.modifiers.sneak) && !Boolean(state.modifiers.sprint)
    const nowMs = now
    const jumpActive = Boolean(state.modifiers.jump) &&
      (Boolean(state.modifiers.keepJump) || nowMs <= state.jumpReleaseAt)

    applyControl('sprint', shouldSprint)
    applyControl('sneak', shouldSneak)
    applyControl('jump', jumpActive)

    if (!jumpActive && state.activeControls.jump) {
      applyControl('jump', false)
    }
  }

  async function pulse(vector = {}, duration = 350, options = {}) {
    const token = Symbol('movement-pulse')
    setTarget(vector, { ...options, jumpHoldMs: options?.jumpHoldMs ?? duration }, token)
    update()
    try {
      await sleep(duration)
    } finally {
      if (clearTarget(token)) {
        update()
      }
    }
  }

  function hold(vector = {}, options = {}) {
    const token = Symbol('movement-hold')
    setTarget(vector, options, token)
    update()
    return ({ immediate = false } = {}) => {
      if (clearTarget(token, { immediate })) {
        update()
      }
    }
  }

  function release({ immediate = false } = {}) {
    clearTarget(null, { immediate })
    update()
  }

  function reset() {
    state.current.forward = 0
    state.current.strafe = 0
    clearTarget(null, { immediate: true })
    for (const key of Object.keys(state.activeControls)) {
      if (state.activeControls[key]) {
        state.activeControls[key] = false
        try {
          state.context?.bot?.setControlState?.(key, false)
        } catch (err) {
          console.warn(`[${label(state.context)}] Failed to reset control ${key}:`, err?.message ?? err)
        }
      }
    }
  }

  return {
    update,
    pulse,
    hold,
    release,
    reset
  }
}

function ensureMovementController(context) {
  if (!context) return null
  if (!context.movementController) {
    context.movementController = createMovementController(context)
  }
  return context.movementController
}

async function smoothMovementPulse(context, vector = {}, duration = 350, options = {}) {
  const controller = ensureMovementController(context)
  if (controller) {
    await controller.pulse(vector, duration, options)
    return
  }
  const bot = context?.bot
  if (!bot) {
    await sleep(duration)
    return
  }
  const states = []
  const forward = Number(vector.forward ?? 0)
  const strafe = Number(vector.strafe ?? 0)
  if (forward > 0) states.push('forward')
  if (forward < 0) states.push('back')
  if (strafe > 0) states.push('right')
  if (strafe < 0) states.push('left')
  if (options.sprint) states.push('sprint')
  if (options.sneak) states.push('sneak')
  if (options.jump) states.push('jump')
  for (const state of states) {
    bot.setControlState(state, true)
  }
  await sleep(duration)
  for (const state of states) {
    bot.setControlState(state, false)
  }
}

function holdMovement(context, vector = {}, options = {}) {
  const controller = ensureMovementController(context)
  if (controller) {
    return controller.hold(vector, options)
  }
  const bot = context?.bot
  if (!bot) return () => {}
  const states = []
  const forward = Number(vector.forward ?? 0)
  const strafe = Number(vector.strafe ?? 0)
  if (forward > 0) states.push('forward')
  if (forward < 0) states.push('back')
  if (strafe > 0) states.push('right')
  if (strafe < 0) states.push('left')
  if (options.sprint) states.push('sprint')
  if (options.sneak) states.push('sneak')
  if (options.jump) states.push('jump')
  for (const state of states) {
    bot.setControlState(state, true)
  }
  return ({ immediate = false } = {}) => {
    for (const state of states) {
      bot.setControlState(state, false)
    }
  }
}

function releaseMovement(context, options = {}) {
  const controller = ensureMovementController(context)
  if (controller) {
    controller.release(options)
  }
}

async function smoothLookTo(context, targetYaw, targetPitch, options = {}) {
  const bot = context?.bot
  if (!bot?.entity) return
  const currentYaw = bot.entity.yaw ?? 0
  const currentPitch = bot.entity.pitch ?? 0
  const yawDelta = normalizeAngle(targetYaw - currentYaw)
  const clampedPitch = clampNumber(targetPitch, -Math.PI / 2, Math.PI / 2)
  const steps = Math.max(1, Math.floor(Number.isFinite(options.steps) ? options.steps : LOOK_SMOOTH_STEPS))
  const duration = Math.max(0, Number.isFinite(options.duration) ? options.duration : LOOK_SMOOTH_DURATION_MS)
  const stepDelay = steps > 0 ? duration / steps : 0

  for (let i = 1; i <= steps; i++) {
    const t = i / steps
    const nextYaw = currentYaw + yawDelta * t
    const nextPitch = currentPitch + (clampedPitch - currentPitch) * t
    try {
      await bot.look(nextYaw, nextPitch, true)
    } catch (err) {
      console.warn(`[${label(context)}] Smooth look failed:`, err?.message ?? err)
      break
    }
    if (stepDelay > 0 && i < steps) {
      await sleep(stepDelay)
    }
  }
}

async function smoothLookBy(context, deltaYaw = 0, deltaPitch = 0, options = {}) {
  const bot = context?.bot
  if (!bot?.entity) return
  const currentYaw = bot.entity.yaw ?? 0
  const currentPitch = bot.entity.pitch ?? 0
  const targetYaw = currentYaw + deltaYaw
  const targetPitch = currentPitch + deltaPitch
  await smoothLookTo(context, targetYaw, targetPitch, options)
}

let movementUpdateTimer = null

function startMovementUpdateLoop() {
  if (movementUpdateTimer) return
  movementUpdateTimer = setInterval(() => {
    const now = monotonicNow()
    for (const ctx of contexts) {
      try {
        ctx?.movementController?.update(now)
      } catch (err) {
        console.warn(`[${label(ctx)}] Movement smoothing update failed:`, err?.message ?? err)
      }
    }
  }, MOVEMENT_UPDATE_INTERVAL_MS)
}

function stopMovementUpdateLoop() {
  if (movementUpdateTimer) {
    clearInterval(movementUpdateTimer)
    movementUpdateTimer = null
  }
}

startMovementUpdateLoop()

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

function romanNumeral(value) {
  if (!Number.isFinite(value) || value <= 0) {
    return `${Math.max(1, Math.floor(Math.abs(value) || 1))}`
  }
  const numerals = [
    [1000, 'M'],
    [900, 'CM'],
    [500, 'D'],
    [400, 'CD'],
    [100, 'C'],
    [90, 'XC'],
    [50, 'L'],
    [40, 'XL'],
    [10, 'X'],
    [9, 'IX'],
    [5, 'V'],
    [4, 'IV'],
    [1, 'I']
  ]
  let remaining = Math.floor(value)
  let output = ''
  for (const [step, glyph] of numerals) {
    while (remaining >= step) {
      output += glyph
      remaining -= step
    }
  }
  return output || 'I'
}

function ensureLineageRecord(name) {
  const lineageName = name || LINEAGE_ROOT_NAME
  let stats = lineageStats.get(lineageName)
  if (!stats) {
    stats = {
      createdAtGeneration: baselineState.generation,
      survivalStreak: 0,
      prestige: 0,
      births: 0,
      lastSeenGeneration: baselineState.generation,
      dominanceStreak: 0,
      lastDominanceGeneration: -Infinity,
      lastDominanceMitigation: -Infinity,
      diversificationCount: 0,
      instability: 0
    }
    lineageStats.set(lineageName, stats)
  } else {
    if (typeof stats.dominanceStreak !== 'number') stats.dominanceStreak = 0
    if (!Number.isFinite(stats.lastDominanceGeneration)) stats.lastDominanceGeneration = -Infinity
    if (!Number.isFinite(stats.lastDominanceMitigation)) stats.lastDominanceMitigation = -Infinity
    if (!Number.isFinite(stats.diversificationCount)) stats.diversificationCount = 0
    if (!Number.isFinite(stats.instability)) stats.instability = 0
  }
  return stats
}

function extractLineageBase(name) {
  if (typeof name !== 'string' || !name) return ''
  const normalized = name.trim()
  const hyphenIndex = normalized.indexOf('-')
  if (hyphenIndex > 0) {
    return normalized.slice(0, hyphenIndex)
  }
  return normalized
}

function randomElement(list, fallback = '') {
  if (!Array.isArray(list) || list.length === 0) {
    return fallback
  }
  const idx = Math.floor(Math.random() * list.length)
  const value = list[idx]
  return value == null ? fallback : value
}

function generateRandomLineageBase() {
  const prefix = randomElement(PREFIXES, '')
  const core = randomElement(ROCK_PARTS, 'Basalt')
  const suffix = randomElement(SUFFIXES, '')
  let base = `${prefix ?? ''}${core ?? ''}${suffix ?? ''}`
  if (Math.random() < 0.35) {
    const extra = randomElement(ROCK_PARTS, '')
    if (extra) {
      base += extra
    }
  }
  base = sanitizeNetworkString(base, {
    fallback: LINEAGE_ROOT_NAME,
    maxLength: Math.max(4, MAX_USERNAME_LENGTH - 2),
    allowed: /[0-9A-Za-z_\-]/,
    label: 'lineage'
  })
  if (!base) {
    base = `Lineage${Math.floor(Math.random() * 10000)}`
  }
  return base
}

function combineLineageBases(primary, partner) {
  const baseA = extractLineageBase(primary) || generateRandomLineageBase()
  const baseB = extractLineageBase(partner)
  if (!baseB || baseB === baseA) {
    return baseA
  }
  const pivotA = Math.max(2, Math.floor(baseA.length / 2))
  const pivotB = Math.max(2, Math.ceil(baseB.length / 2))
  let combined = `${baseA.slice(0, pivotA)}${baseB.slice(Math.max(0, pivotB - 1))}`
  if (!combined || combined.length < 4) {
    combined = `${baseA}${baseB}`
  }
  combined = sanitizeNetworkString(combined, {
    fallback: `${baseA}${baseB}`.slice(0, MAX_USERNAME_LENGTH),
    maxLength: Math.max(4, MAX_USERNAME_LENGTH - 1),
    allowed: /[0-9A-Za-z_\-]/,
    label: 'lineage'
  })
  if (!combined) {
    combined = generateRandomLineageBase()
  }
  return combined
}

function deriveLineageVariantBase(base, stats = null) {
  const root = extractLineageBase(base) || generateRandomLineageBase()
  let counter = stats ? stats.diversificationCount ?? 0 : 0
  const suffix = LINEAGE_VARIANT_SUFFIXES[counter % LINEAGE_VARIANT_SUFFIXES.length] ?? 'Variant'
  let variant = `${root}${suffix}`
  if (counter >= LINEAGE_VARIANT_SUFFIXES.length) {
    variant += `${counter + 1}`
  }
  variant = sanitizeNetworkString(variant, {
    fallback: `${root}${suffix}`.slice(0, MAX_USERNAME_LENGTH - 1),
    maxLength: Math.max(4, MAX_USERNAME_LENGTH - 1),
    allowed: /[0-9A-Za-z_\-]/,
    label: 'lineage'
  })
  if (!variant) {
    variant = generateRandomLineageBase()
  }
  if (stats) {
    stats.diversificationCount = (stats.diversificationCount ?? 0) + 1
  }
  return variant
}

function formatLineageName(base, ordinal) {
  if (ordinal <= 1) {
    return base
  }
  const suffix = `-${romanNumeral(ordinal)}`
  let workingBase = base
  let candidate = `${workingBase}${suffix}`
  if (candidate.length > MAX_USERNAME_LENGTH) {
    const overflow = candidate.length - MAX_USERNAME_LENGTH
    workingBase = workingBase.slice(0, Math.max(3, workingBase.length - overflow))
    candidate = `${workingBase}${suffix}`
    if (candidate.length > MAX_USERNAME_LENGTH) {
      candidate = `${workingBase}${suffix.replace('-', '')}`
    }
    if (candidate.length > MAX_USERNAME_LENGTH) {
      candidate = candidate.slice(0, MAX_USERNAME_LENGTH)
    }
  }
  return candidate
}

function allocateLineageIdentity({ parentLineage = LINEAGE_ROOT_NAME, mode = 'civilized' } = {}) {
  const lineage = parentLineage || LINEAGE_ROOT_NAME
  ensureLineageRecord(lineage)
  let ordinal = (lineageCounters.get(lineage) ?? 0) + 1
  let attempt = 0
  while (attempt < 50) {
    const rawName = formatLineageName(lineage, ordinal)
    const sanitized = sanitizeNetworkString(rawName, {
      fallback: rawName,
      maxLength: MAX_USERNAME_LENGTH,
      allowed: /[0-9A-Za-z_\-]/,
      label: 'username'
    })
    if (sanitized && !usedNames.has(sanitized)) {
      usedNames.add(sanitized)
      lineageCounters.set(lineage, ordinal)
      const stats = ensureLineageRecord(lineage)
      stats.births = (stats.births ?? 0) + 1
      stats.lastSeenGeneration = baselineState.generation
      return { username: sanitized, lineage, ordinal }
    }
    ordinal += 1
    attempt += 1
  }
  const fallback = sanitizeNetworkString(`BrainBot${Math.floor(Math.random() * 100000)}`, {
    fallback: 'BrainBot',
    maxLength: MAX_USERNAME_LENGTH,
    allowed: /[0-9A-Za-z_\-]/,
    label: 'username'
  })
  usedNames.add(fallback)
  const stats = ensureLineageRecord(lineage)
  stats.births = (stats.births ?? 0) + 1
  stats.lastSeenGeneration = baselineState.generation
  return { username: fallback, lineage, ordinal }
}

function generateRandomName(options = {}) {
  const { username } = allocateLineageIdentity(options)
  return username
}

function label(context) {
  if (!context) return 'unknown'
  return context.mode === 'feral' ? `${context.username}[F]` : `${context.username}`
}

async function initializeBaselineBrain() {
  try {
    const loaded = await loadBrain(CHECKPOINT_DIR, OBS_SIZE, ACTIONS.length)
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
    if (isRemoteBrainUnavailableError(err)) {
      throw err
    }
    console.error('[Baseline] Failed to initialize from checkpoint:', err)
    try {
      baselineBrain = await createBrain(OBS_SIZE, ACTIONS.length)
      if (baselineBrain) {
        baselineBrain.owner = 'hivemind'
      }
    } catch (creationError) {
      if (isRemoteBrainUnavailableError(creationError)) {
        throw creationError
      }
      console.error('[Baseline] Failed to create baseline brain after initialization error:', creationError)
      throw creationError
    }
  }

  markBaselineWeightsHealthy('init')
  return baselineBrain
}

async function ensureBaselineReady() {
  if (baselineBrain) return baselineBrain
  if (!baselineReady) {
    baselineReady = initializeBaselineBrain()
  }
  try {
    await baselineReady
  } catch (err) {
    if (isRemoteBrainUnavailableError(err)) {
      baselineReady = null
      throw err
    }
    console.error('[Baseline] Initialization failed, recreating model:', err)
    try {
      baselineBrain = await createBrain(OBS_SIZE, ACTIONS.length)
      if (baselineBrain) {
        baselineBrain.owner = 'hivemind'
      }
      baselineReady = Promise.resolve(baselineBrain)
    } catch (creationError) {
      if (isRemoteBrainUnavailableError(creationError)) {
        baselineReady = null
        throw creationError
      }
      console.error('[Baseline] Failed to recreate baseline brain:', creationError)
      baselineReady = null
      throw creationError
    }
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
    if (isRemoteBrainUnavailableError(err)) {
      throw err
    }
    console.error(`[Baseline] Failed to save checkpoint (${reason}):`, err)
  }
}

function scheduleBaselineSave(reason = 'periodic') {
  if (saveInFlight) {
    pendingSaveReason = reason
    return
  }

  if (!isRemoteBrainConnected()) {
    if (!deferredBaselineSaveReason) {
      deferredBaselineSaveReason = reason
    }
    const now = Date.now()
    if (now >= nextBaselineSaveLogAt) {
      const status = getRemoteBrainStatus()
      console.warn(`[Baseline] Deferring save until remote brain returns (${describeRemoteRetry(status)}).`)
      nextBaselineSaveLogAt = now + 5000
    }
    return
  }

  deferredBaselineSaveReason = null
  nextBaselineSaveLogAt = 0

  saveInFlight = (async () => {
    await persistBaseline(reason)
  })()

  saveInFlight
    .catch(err => {
      if (isRemoteBrainUnavailableError(err)) {
        if (!deferredBaselineSaveReason) {
          deferredBaselineSaveReason = reason
        }
        const status = getRemoteBrainStatus()
        console.warn(`[Baseline] Save deferred: remote brain unavailable (${describeRemoteRetry(status)}).`)
      } else {
        console.error('[Baseline] Save task error:', err)
      }
    })
    .finally(() => {
      saveInFlight = null
      if (pendingSaveReason) {
        const nextReason = pendingSaveReason
        pendingSaveReason = null
        scheduleBaselineSave(nextReason)
      } else if (deferredBaselineSaveReason && isRemoteBrainConnected()) {
        const nextReason = deferredBaselineSaveReason
        deferredBaselineSaveReason = null
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

async function safePersistState(reason = 'recovery') {
  try {
    await flushPendingSave()
  } catch (err) {
    console.error(`[Baseline] Failed to flush pending save during ${reason}:`, err)
  }

  if (!isRemoteBrainConnected()) {
    const status = getRemoteBrainStatus()
    console.warn(
      `[Baseline] Remote brain unavailable during ${reason} save (${describeRemoteRetry(status)}).`
    )
    return
  }

  try {
    await ensureBaselineReady()
  } catch (err) {
    if (isRemoteBrainUnavailableError(err)) {
      const status = getRemoteBrainStatus()
      console.warn(
        `[Baseline] Baseline unavailable for ${reason} save (${describeRemoteRetry(status)}).`
      )
      return
    }
    console.error(`[Baseline] Failed to prepare baseline for ${reason} save:`, err)
    return
  }

  try {
    await persistBaseline(reason)
  } catch (err) {
    if (isRemoteBrainUnavailableError(err)) {
      const status = getRemoteBrainStatus()
      console.warn(
        `[Baseline] Skipping ${reason} save: remote brain unavailable (${describeRemoteRetry(status)}).`
      )
    } else {
      console.error(`[Baseline] Failed to persist baseline during ${reason}:`, err)
    }
  }
}

async function ensureContextBrain(context) {
  await ensureBaselineReady()
  if (!baselineBrain) {
    throw new Error('Baseline brain failed to initialize')
  }

  if (context.brain && context.brain.id) {
    context.brain.owner = label(context)
    return context.brain
  }

  if (context.waitingForBrain) {
    return context.waitingForBrain
  }

  context.waitingForBrain = (async () => {
    const brain = await createBrain(OBS_SIZE, ACTIONS.length)
    brain.owner = label(context)

    if (baselineBrain && baselineBrain.id && brain.id && brain.id !== baselineBrain.id) {
      try {
        await copyWeights(brain, baselineBrain)
      } catch (err) {
        if (isRemoteBrainUnavailableError(err)) {
          throw err
        }
        console.error(`[${label(context)}] Failed to seed brain from baseline:`, err)
      }
    }

    if (!Array.isArray(context.pendingMutations)) {
      context.pendingMutations = []
    }

    while (context.pendingMutations.length) {
      const stddev = context.pendingMutations.shift()
      if (!Number.isFinite(stddev) || stddev <= 0) {
        continue
      }
      try {
        await mutateWeights(brain, Math.min(NEW_BRAIN_MUTATION_MAX, Math.max(0, stddev)))
      } catch (err) {
        if (isRemoteBrainUnavailableError(err)) {
          context.pendingMutations.unshift(stddev)
          throw err
        }
        console.error(`[${label(context)}] Failed to mutate fresh brain:`, err)
      }
    }

    context.brain = brain
    markContextWeightsHealthy(context, 'init')
    return brain
  })()

  try {
    return await context.waitingForBrain
  } finally {
    context.waitingForBrain = null
  }
}

function computeNoveltyKey(vector) {
  const values = Array.from(vector.slice(0, 16)).map(value => {
    if (!Number.isFinite(value)) return 0
    const precision = Math.pow(10, NOVELTY_HASH_PRECISION)
    return Math.round(value * precision) / precision
  })
  return values.join('|')
}

function updateBehaviorEntropy(context, action) {
  const counts = context.actionCounts
  counts.set(action, (counts.get(action) ?? 0) + 1)
  context.actionHistory.push(action)
  if (context.actionHistory.length > DIVERSITY_WINDOW) {
    const removed = context.actionHistory.shift()
    const prev = counts.get(removed) ?? 0
    if (prev <= 1) {
      counts.delete(removed)
    } else {
      counts.set(removed, prev - 1)
    }
  }

  const total = Array.from(counts.values()).reduce((sum, value) => sum + value, 0)
  if (total > 0) {
    let entropy = 0
    for (const value of counts.values()) {
      const p = value / total
      entropy -= p * Math.log2(p)
    }
    const maxEntropy = Math.log2(ACTIONS.length)
    context.behaviorEntropy = maxEntropy > 0 ? entropy / maxEntropy : 0
  } else {
    context.behaviorEntropy = 0
  }
}

function updateSkillChains(context, reward) {
  if (!context.actionHistory?.length) return
  const historyLength = Math.min(4, context.actionHistory.length)
  const sequence = context.actionHistory.slice(-historyLength).join('>')
  if (!sequence) return
  const record = context.skillChains.get(sequence) ?? { count: 0, total: 0 }
  record.count += 1
  record.total += reward
  context.skillChains.set(sequence, record)
  const average = record.total / record.count
  context.currentChainScore = Math.max(0, Math.min(1, average))
}

function limitMagnitude(value, limit) {
  if (!Number.isFinite(value)) return 0
  if (!Number.isFinite(limit) || limit <= 0) return value
  if (value > limit) return limit
  if (value < -limit) return -limit
  return value
}

function limitPositive(value, limit) {
  if (!Number.isFinite(value) || value <= 0) return 0
  if (!Number.isFinite(limit) || limit <= 0) return Math.max(0, value)
  return Math.min(limit, value)
}

function normalizeMagnitude(value, limit) {
  if (!Number.isFinite(value) || !Number.isFinite(limit) || limit <= 0) return 0
  const scaled = value / limit
  if (scaled > 1) return 1
  if (scaled < -1) return -1
  return scaled
}

function normalizePositive(value, limit) {
  if (!Number.isFinite(value) || value <= 0 || !Number.isFinite(limit) || limit <= 0) return 0
  const scaled = value / limit
  if (scaled > 1) return 1
  if (scaled < 0) return 0
  return scaled
}

function sanitizeCommunicationPayload(payload) {
  if (!payload || typeof payload !== 'object') return null
  const result = {}
  for (const [key, value] of Object.entries(payload)) {
    if (typeof value === 'number') {
      if (Number.isFinite(value)) {
        result[key] = value
      }
    } else if (typeof value === 'string') {
      if (value) {
        result[key] = value.slice(0, 64)
      }
    } else if (typeof value === 'boolean') {
      result[key] = value
    }
  }
  return Object.keys(result).length ? result : null
}

function createCommunicationState() {
  return {
    lastSentAt: 0,
    lastHeardAt: 0,
    nextAvailableAt: 0,
    totalSent: 0,
    sentCounts: Object.create(null),
    summary: {
      allies: new Float32Array(COMMUNICATION_TYPE_COUNT),
      others: new Float32Array(COMMUNICATION_TYPE_COUNT),
      alliesCount: new Uint16Array(COMMUNICATION_TYPE_COUNT),
      othersCount: new Uint16Array(COMMUNICATION_TYPE_COUNT),
      direction: { x: 0, z: 0 },
      updatedAt: 0
    }
  }
}

function ensureCommunicationState(context) {
  if (!context) return null
  if (!context.communication) {
    context.communication = createCommunicationState()
  }
  const summary = context.communication.summary
  if (!(summary.allies instanceof Float32Array) || summary.allies.length !== COMMUNICATION_TYPE_COUNT) {
    summary.allies = new Float32Array(COMMUNICATION_TYPE_COUNT)
  }
  if (!(summary.others instanceof Float32Array) || summary.others.length !== COMMUNICATION_TYPE_COUNT) {
    summary.others = new Float32Array(COMMUNICATION_TYPE_COUNT)
  }
  if (!(summary.alliesCount instanceof Uint16Array) || summary.alliesCount.length !== COMMUNICATION_TYPE_COUNT) {
    summary.alliesCount = new Uint16Array(COMMUNICATION_TYPE_COUNT)
  }
  if (!(summary.othersCount instanceof Uint16Array) || summary.othersCount.length !== COMMUNICATION_TYPE_COUNT) {
    summary.othersCount = new Uint16Array(COMMUNICATION_TYPE_COUNT)
  }
  if (!summary.direction) {
    summary.direction = { x: 0, z: 0 }
  }
  return context.communication
}

function pruneCommunicationBus(now = Date.now()) {
  const bus = COMMUNICATION_BUS
  if (!bus) return
  if (now < bus.lastPrune + COMMUNICATION_PRUNE_INTERVAL_MS) {
    return
  }
  bus.lastPrune = now
  const cutoff = now - COMMUNICATION_DECAY_MS
  if (!Number.isFinite(cutoff) || cutoff <= 0) {
    return
  }
  if (!Array.isArray(bus.messages) || bus.messages.length === 0) {
    bus.messages = []
    return
  }
  const filtered = []
  for (const message of bus.messages) {
    const createdAt = Number(message?.createdAt)
    if (!Number.isFinite(createdAt) || createdAt < cutoff) {
      continue
    }
    filtered.push(message)
  }
  if (filtered.length > COMMUNICATION_MESSAGE_LIMIT) {
    bus.messages = filtered.slice(filtered.length - COMMUNICATION_MESSAGE_LIMIT)
  } else {
    bus.messages = filtered
  }
}

function broadcastCommunication(context, type, intensity = 1, payload = null) {
  if (!context || !COMMUNICATION_TYPE_INDEX.has(type)) {
    return null
  }
  const comm = ensureCommunicationState(context)
  if (!comm) return null
  const now = Date.now()
  pruneCommunicationBus(now)
  const sanitizedIntensity = limitPositive(Math.abs(Number(intensity)) || 0, COMMUNICATION_INTENSITY_CLAMP)
  if (sanitizedIntensity <= 0) {
    return null
  }
  const position = context.bot?.entity?.position
  const record = {
    senderId: context.id,
    lineage: context.lineage ?? null,
    mode: context.mode ?? null,
    type,
    intensity: sanitizedIntensity,
    createdAt: now,
    position: position
      ? {
          x: Number(position.x) || 0,
          y: Number(position.y) || 0,
          z: Number(position.z) || 0
        }
      : null,
    payload: sanitizeCommunicationPayload(payload)
  }
  COMMUNICATION_BUS.messages.push(record)
  if (COMMUNICATION_BUS.messages.length > COMMUNICATION_MESSAGE_LIMIT) {
    COMMUNICATION_BUS.messages.splice(
      0,
      Math.max(0, COMMUNICATION_BUS.messages.length - COMMUNICATION_MESSAGE_LIMIT)
    )
  }
  comm.lastSentAt = now
  comm.nextAvailableAt = now + COMMUNICATION_COOLDOWN_MS
  comm.totalSent += 1
  comm.sentCounts[type] = (comm.sentCounts[type] ?? 0) + 1
  return record
}

function computeCommunicationIntensity(context, type) {
  switch (type) {
    case 'resource': {
      const inventory = Number.isFinite(context.lastInvTotal) ? context.lastInvTotal : 0
      return Math.min(COMMUNICATION_INTENSITY_CLAMP, 1 + inventory / 32)
    }
    case 'danger': {
      const health = Number(context?.bot?.health ?? 20)
      return Math.min(COMMUNICATION_INTENSITY_CLAMP, 1 + Math.max(0, (20 - health) / 5))
    }
    case 'assist': {
      const repetition = Number.isFinite(context.repetitionStreak) ? context.repetitionStreak : 0
      return Math.min(COMMUNICATION_INTENSITY_CLAMP, 1 + repetition / 6)
    }
    case 'gather': {
      const resources = context.resources ?? {}
      const focus = Math.max(0, resources.ore ?? 0, resources.wood ?? 0, resources.stone ?? 0)
      return Math.min(COMMUNICATION_INTENSITY_CLAMP, 1 + focus / 16)
    }
    case 'status': {
      const morale = Number.isFinite(context?.morale?.value) ? context.morale.value : MORALE_BASELINE
      return Math.min(COMMUNICATION_INTENSITY_CLAMP, 1 + Math.abs(morale - MORALE_BASELINE) * 2)
    }
    default:
      return Math.min(COMMUNICATION_INTENSITY_CLAMP, 1)
  }
}

function deriveCommunicationPayload(context, type) {
  switch (type) {
    case 'resource': {
      const resources = context.resources ?? {}
      const entries = Object.entries(resources)
      entries.sort((a, b) => Number(b[1] ?? 0) - Number(a[1] ?? 0))
      const top = entries[0]?.[0] ?? null
      return {
        focus: top,
        total: Number.isFinite(context.lastInvTotal) ? context.lastInvTotal : 0
      }
    }
    case 'danger':
      return {
        health: Number(context?.bot?.health ?? 20),
        food: Number(context?.bot?.food ?? 20),
        oxygen: Number(context?.bot?.oxygenLevel ?? context?.bot?.oxygen ?? 20)
      }
    case 'assist':
      return {
        mode: context.mode,
        stagnation: context.stagnation?.active ?? false,
        chain: context.currentChainScore ?? 0
      }
    case 'gather':
      return {
        novelty: context.noveltyFlag ?? false,
        diversity: GLOBAL_RESOURCE_POOL.diversity.size
      }
    case 'status':
      return {
        morale: Number.isFinite(context?.morale?.value) ? context.morale.value : MORALE_BASELINE,
        epsilon: Number.isFinite(context?.epsilon) ? context.epsilon : EPSILON_START
      }
    default:
      return null
  }
}

function updateCommunicationAwareness(context) {
  const comm = ensureCommunicationState(context)
  if (!comm) return
  const summary = comm.summary
  summary.allies.fill(0)
  summary.others.fill(0)
  summary.alliesCount.fill(0)
  summary.othersCount.fill(0)
  let directionX = 0
  let directionZ = 0
  let directionWeight = 0
  const now = Date.now()
  pruneCommunicationBus(now)
  const position = context.bot?.entity?.position ?? null
  let lastHeard = comm.lastHeardAt || 0

  for (const message of COMMUNICATION_BUS.messages) {
    if (!message || !COMMUNICATION_TYPE_INDEX.has(message.type)) continue
    const dt = now - Number(message.createdAt)
    if (!Number.isFinite(dt) || dt < 0 || dt > COMMUNICATION_DECAY_MS) continue
    let intensity = limitPositive(Number(message.intensity) || 0, COMMUNICATION_INTENSITY_CLAMP)
    if (intensity <= 0) continue
    const decay = 1 - dt / COMMUNICATION_DECAY_MS
    if (decay <= 0) continue
    intensity *= decay
    let withinRange = true
    let dx = 0
    let dz = 0
    if (position && message.position) {
      dx = Number(message.position.x) - Number(position.x)
      const dy = Number(message.position.y) - Number(position.y)
      dz = Number(message.position.z) - Number(position.z)
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz)
      if (!Number.isFinite(dist) || dist > COMMUNICATION_RANGE) {
        withinRange = false
      } else {
        const falloff = Math.max(0, 1 - dist / COMMUNICATION_RANGE)
        intensity *= falloff
      }
    }
    if (!withinRange || intensity <= 0) continue
    const index = COMMUNICATION_TYPE_INDEX.get(message.type)
    if (index == null) continue
    if (message.lineage && context.lineage && message.lineage === context.lineage) {
      summary.allies[index] += intensity
      summary.alliesCount[index] += 1
      if (position && message.position) {
        directionX += dx * intensity
        directionZ += dz * intensity
        directionWeight += intensity
      }
    } else {
      summary.others[index] += intensity
      summary.othersCount[index] += 1
    }
    if (message.senderId !== context.id) {
      lastHeard = Math.max(lastHeard, Number(message.createdAt))
    }
  }

  summary.direction.x = directionWeight > 0 ? directionX / directionWeight : 0
  summary.direction.z = directionWeight > 0 ? directionZ / directionWeight : 0
  summary.updatedAt = now
  comm.lastHeardAt = lastHeard
}

async function performCommunicationAction(context, signalType) {
  const type = typeof signalType === 'string' ? signalType.toLowerCase() : ''
  if (!COMMUNICATION_TYPE_INDEX.has(type)) {
    addBlockReward(context, COMMUNICATION_COOLDOWN_PENALTY, REWARD_SIGN.NEGATIVE, 'communication-invalid')
    return false
  }
  const comm = ensureCommunicationState(context)
  if (!comm) {
    return false
  }
  const now = Date.now()
  if (now < comm.nextAvailableAt) {
    addBlockReward(context, COMMUNICATION_COOLDOWN_PENALTY, REWARD_SIGN.NEGATIVE, `communication-cooldown-${type}`)
    return false
  }

  const intensity = computeCommunicationIntensity(context, type)
  const payload = deriveCommunicationPayload(context, type)
  const record = broadcastCommunication(context, type, intensity, payload)
  if (record) {
    console.log(`[${label(context)}] Broadcast ${type} signal (intensity=${record.intensity.toFixed(2)}).`)
    addBlockReward(context, COMMUNICATION_REWARD, REWARD_SIGN.POSITIVE, `communication-${type}`)
    return true
  }
  addBlockReward(context, COMMUNICATION_COOLDOWN_PENALTY, REWARD_SIGN.NEGATIVE, `communication-failed-${type}`)
  return false
}

function ensureTemporalMemoryState(context) {
  if (!context) return
  if (!Array.isArray(context.memoryRewards)) {
    context.memoryRewards = []
  }
  if (!Array.isArray(context.memoryActions)) {
    context.memoryActions = []
  }
  if (!(context.actionValueMemory instanceof Map)) {
    context.actionValueMemory = new Map()
  }
  if (!Number.isFinite(context.recentRewardAvg)) {
    context.recentRewardAvg = 0
  }
  if (!Number.isFinite(context.longRewardAvg)) {
    context.longRewardAvg = 0
  }
  if (!Number.isFinite(context.rewardDrift)) {
    context.rewardDrift = 0
  }
  if (!Number.isFinite(context.rewardVolatility)) {
    context.rewardVolatility = 0
  }
  if (!Number.isFinite(context.actionMemoryBest)) {
    context.actionMemoryBest = 0
  }
  if (!Number.isFinite(context.actionMemoryWorst)) {
    context.actionMemoryWorst = 0
  }
  if (!Number.isFinite(context.damageDebt)) {
    context.damageDebt = 0
  }
  if (!Number.isFinite(context.recentDamage)) {
    context.recentDamage = 0
  }
}

function pushBounded(list, value, limit) {
  if (!Array.isArray(list)) return
  list.push(value)
  if (list.length > limit) {
    list.splice(0, list.length - limit)
  }
}

function updateTemporalMemory(context, reward) {
  ensureTemporalMemoryState(context)
  const rewardValue = Number.isFinite(reward) ? reward : 0
  pushBounded(context.memoryRewards, rewardValue, MEMORY_REWARD_WINDOW)

  const shortDecay = MEMORY_SHORT_DECAY
  const longDecay = MEMORY_LONG_DECAY
  const prevShort = context.recentRewardAvg
  const prevLong = context.longRewardAvg
  const newShort = prevShort * (1 - shortDecay) + rewardValue * shortDecay
  const newLong = prevLong * (1 - longDecay) + rewardValue * longDecay
  context.recentRewardAvg = limitMagnitude(newShort, REWARD_MEMORY_CLAMP)
  context.longRewardAvg = limitMagnitude(newLong, REWARD_MEMORY_CLAMP)
  context.rewardDrift = limitMagnitude(context.recentRewardAvg - context.longRewardAvg, REWARD_MEMORY_CLAMP)

  const prevVolatility = context.rewardVolatility
  const volatilityDelta = Math.abs(rewardValue - context.recentRewardAvg)
  const newVolatility = prevVolatility * (1 - MEMORY_VOLATILITY_DECAY) + volatilityDelta * MEMORY_VOLATILITY_DECAY
  context.rewardVolatility = limitPositive(newVolatility, REWARD_MEMORY_CLAMP)

  const retention = 1 - ACTION_MEMORY_RATE
  const entries = Array.from(context.actionValueMemory.entries())
  for (const [key, entry] of entries) {
    const base = Number.isFinite(entry?.value) ? entry.value : 0
    const decayed = limitMagnitude(base * retention, ACTION_VALUE_CLAMP)
    if (Math.abs(decayed) <= ACTION_MEMORY_EPSILON) {
      context.actionValueMemory.delete(key)
    } else {
      entry.value = decayed
      context.actionValueMemory.set(key, entry)
    }
  }

  if (Number.isInteger(context.lastAction)) {
    const key = context.lastAction
    const entry = context.actionValueMemory.get(key) ?? { value: 0 }
    const updated = limitMagnitude(entry.value + rewardValue * ACTION_MEMORY_RATE, ACTION_VALUE_CLAMP)
    entry.value = updated
    context.actionValueMemory.set(key, entry)
  }

  let best = -Infinity
  let worst = Infinity
  for (const entry of context.actionValueMemory.values()) {
    const value = Number.isFinite(entry?.value) ? entry.value : 0
    if (value > best) best = value
    if (value < worst) worst = value
  }
  context.actionMemoryBest = best === -Infinity ? 0 : limitMagnitude(best, ACTION_VALUE_CLAMP)
  context.actionMemoryWorst = worst === Infinity ? 0 : limitMagnitude(worst, ACTION_VALUE_CLAMP)
}

function recordActionMemory(context, actionIndex) {
  if (!Number.isInteger(actionIndex)) return
  ensureTemporalMemoryState(context)
  pushBounded(context.memoryActions, actionIndex, MEMORY_ACTION_WINDOW)
}

function populateMemoryObservation(context, obs, startIndex) {
  ensureTemporalMemoryState(context)
  let index = startIndex

  const rewards = context.memoryRewards
  for (let i = 0; i < MEMORY_REWARD_WINDOW; i++) {
    const sourceIndex = rewards.length - 1 - i
    const value = sourceIndex >= 0 ? rewards[sourceIndex] : 0
    obs[index++] = normalizeMagnitude(value, REWARD_MEMORY_CLAMP)
  }

  const actions = context.memoryActions
  const actionNormalizer = Math.max(1, ACTIONS.length - 1)
  for (let i = 0; i < MEMORY_ACTION_WINDOW; i++) {
    const sourceIndex = actions.length - 1 - i
    const value = sourceIndex >= 0 ? actions[sourceIndex] : -1
    if (!Number.isFinite(value) || value < 0) {
      obs[index++] = 0
    } else {
      obs[index++] = Math.max(0, Math.min(1, value / actionNormalizer))
    }
  }

  obs[index++] = normalizeMagnitude(context.recentRewardAvg, REWARD_MEMORY_CLAMP)
  obs[index++] = normalizeMagnitude(context.longRewardAvg, REWARD_MEMORY_CLAMP)
  obs[index++] = normalizeMagnitude(context.rewardDrift, REWARD_MEMORY_CLAMP)
  obs[index++] = normalizePositive(context.rewardVolatility, REWARD_MEMORY_CLAMP)
  obs[index++] = normalizeMagnitude(context.actionMemoryBest, ACTION_VALUE_CLAMP)
  obs[index++] = normalizeMagnitude(context.actionMemoryWorst, ACTION_VALUE_CLAMP)
  obs[index++] = normalizePositive(context.damageDebt, DAMAGE_MEMORY_CLAMP)
  obs[index++] = normalizePositive(context.recentDamage, DAMAGE_MEMORY_CLAMP)
  const pendingMutations = Array.isArray(context.pendingMutations)
    ? context.pendingMutations.length
    : 0
  obs[index++] = normalizePositive(pendingMutations, MUTATION_QUEUE_CLAMP)
  const epsilonBoost = Number.isFinite(context.epsilonBoost) ? Math.abs(context.epsilonBoost) : 0
  obs[index++] = normalizePositive(epsilonBoost, EPSILON_BOOST_NORMALIZER)
  const stagnation = context.stagnation?.streak ?? 0
  const stagnationWindow = STAGNATION_WINDOW > 0 ? STAGNATION_WINDOW : 1
  const normalizedStagnation = Number.isFinite(stagnation)
    ? Math.max(0, Math.min(1, stagnation / stagnationWindow))
    : 0
  obs[index++] = normalizedStagnation
  obs[index++] = context.waitingForBrain ? 1 : 0
  obs[index++] = context.weightsSuspect ? 1 : 0

  return index
}

function trackEnvironmentAwareness(context) {
  try {
    const { bot } = context
    if (!bot?.entity?.position) return
    const basePos = bot.entity.position
    const below = bot.blockAt(basePos.offset(0, -1, 0))
    const ahead = bot.blockAt(basePos.offset(0, 0, 1))
    if (below?.name) {
      context.visitedBlocks.add(below.name)
    }
    if (ahead?.name) {
      context.visitedBlocks.add(ahead.name)
    }
    const biome = bot.entity?.biome?.name ?? bot.biome?.name
    if (biome) {
      context.visitedBiomes.add(biome)
    }
  } catch (err) {
    console.warn(`[${label(context)}] Failed to sample environment:`, err?.message ?? err)
  }
}

function trackNovelty(context, obs) {
  const key = computeNoveltyKey(obs)
  if (!context.visitedStates.has(key)) {
    context.visitedStates.add(key)
    context.noveltyCount += 1
    context.noveltyFlag = true
  } else {
    context.noveltyFlag = false
  }
}

function isOreName(name) {
  if (!name) return false
  const normalized = String(name).toLowerCase()
  if (normalized.startsWith('raw_')) {
    const suffix = normalized.slice(4)
    const rawType = suffix.split('_')[0]
    if (rawType && RAW_ORE_SUFFIXES.has(rawType)) {
      return true
    }
  }
  return ORE_NAME_KEYWORDS.some(keyword => normalized.includes(keyword))
}

function isOreBlock(block) {
  if (!block) return false
  if (typeof block === 'string') {
    return isOreName(block)
  }
  if (typeof block?.name === 'string') {
    return isOreName(block.name)
  }
  return false
}

function categorizeResource(name) {
  if (!name) return null
  const normalized = String(name).toLowerCase()
  if (normalized.includes('log') || normalized.includes('wood')) return 'wood'
  if (normalized.includes('stone') || normalized.includes('cobblestone') || normalized.includes('gravel')) return 'stone'
  if (isOreName(normalized) || normalized.includes('coal') || normalized.includes('iron')) return 'ore'
  return null
}

function registerContribution(context, amount) {
  context.resourceLedger.contributed += amount
  GLOBAL_RESOURCE_POOL.totalContribution += amount
}

function registerWithdrawal(context, amount) {
  context.resourceLedger.withdrawn += amount
  GLOBAL_RESOURCE_POOL.totalWithdrawal += amount
}

function updateCooperationScore(context) {
  const contribution = context.resourceLedger.contributed
  const withdrawal = context.resourceLedger.withdrawn
  const totalContribution = GLOBAL_RESOURCE_POOL.totalContribution || 1
  const totalWithdrawal = GLOBAL_RESOURCE_POOL.totalWithdrawal || 1
  const fairness = contribution / totalContribution - withdrawal / totalWithdrawal
  context.cooperationScore = fairness
}

function noteResourceDiversity(type) {
  if (!type || !RESOURCE_TYPES.includes(type)) return
  const pool = GLOBAL_RESOURCE_POOL
  if ((pool[type] ?? 0) <= 0) return
  if (!pool.diversity.has(type)) {
    pool.diversity.add(type)
    const bonus = 0.08
    for (const ctx of contexts) {
      addBlockReward(ctx, bonus, REWARD_SIGN.POSITIVE, 'resource-diversity')
    }
    console.log(`[Resource] Diversity bonus unlocked for ${type} (${pool.diversity.size}/${RESOURCE_TYPES.length}).`)
  }
}

function ensureEmotionVector(context) {
  if (!(context.emotion instanceof Float32Array) || context.emotion.length !== EMOTION_VECTOR_SIZE) {
    context.emotion = new Float32Array(EMOTION_VECTOR_SIZE)
    for (let i = 0; i < EMOTION_VECTOR_SIZE; i++) {
      context.emotion[i] = MORALE_BASELINE
    }
  }
  return context.emotion
}

function adjustMorale(context, reward) {
  const accumulator = ensureRewardAccumulator(reward)
  if (!context.morale) {
    context.morale = {
      value: MORALE_BASELINE,
      frustration: 0,
      sharpness: 0.5,
      successStreak: 0,
      failureStreak: 0
    }
  }

  const morale = context.morale
  const bounded = Math.max(-2, Math.min(2, accumulator.total))

  if (bounded > 0.2) {
    morale.value = Math.min(1, morale.value + 0.04 + bounded * 0.02)
    morale.frustration = Math.max(0, morale.frustration - 0.03)
    morale.successStreak = (morale.successStreak ?? 0) + 1
    morale.failureStreak = 0
    morale.sharpness = Math.min(1, (morale.sharpness ?? 0.5) + 0.05)
  } else if (bounded < -0.2) {
    morale.value = Math.max(0, morale.value - 0.05 - Math.abs(bounded) * 0.02)
    morale.frustration = Math.min(1, morale.frustration + 0.04 + Math.abs(bounded) * 0.01)
    morale.failureStreak = (morale.failureStreak ?? 0) + 1
    morale.successStreak = 0
    morale.sharpness = Math.max(0, (morale.sharpness ?? 0.5) - 0.06)
  } else {
    morale.value = morale.value * 0.98 + MORALE_BASELINE * 0.02
    morale.frustration *= 0.96
    morale.sharpness = (morale.sharpness ?? 0.5) * 0.97 + 0.5 * 0.03
    morale.successStreak = Math.max(0, (morale.successStreak ?? 0) - 1)
    morale.failureStreak = Math.max(0, (morale.failureStreak ?? 0) - 1)
  }

  const frustrationPenalty = 1 - Math.min(0.6, morale.frustration * 0.4)
  const moraleBoost = 1 + (morale.value - MORALE_BASELINE) * 0.25
  const sharpnessBoost = 1 + ((morale.sharpness ?? 0.5) - 0.5) * 0.15
  let factor = moraleBoost * frustrationPenalty * sharpnessBoost * rewardProfile.morale
  if (!Number.isFinite(factor)) {
    factor = 1
  }
  let modifiedTotal = accumulator.total * factor
  if (!Number.isFinite(modifiedTotal)) {
    modifiedTotal = accumulator.total
    factor = 1
  }
  const scale = Math.max(0, Number.isFinite(factor) ? Math.abs(factor) : 1)
  accumulator.total = modifiedTotal
  accumulator.reward *= scale
  accumulator.penalty *= scale

  const emotion = ensureEmotionVector(context)
  const targets = [morale.value, morale.frustration, morale.sharpness ?? 0.5]
  for (let i = 0; i < EMOTION_VECTOR_SIZE; i++) {
    emotion[i] = emotion[i] * EMOTION_DECAY + targets[i] * (1 - EMOTION_DECAY)
  }

  return accumulator
}

function mutateRewardProfile() {
  const adjustments = []
  for (const key of Object.keys(rewardProfile)) {
    const jitter = (Math.random() - 0.5) * 0.18
    const next = rewardProfile[key] * (1 + jitter)
    rewardProfile[key] = Math.min(1.8, Math.max(0.4, next))
    adjustments.push(`${key}=${rewardProfile[key].toFixed(2)}`)
  }
  const jitterDelay = Math.floor(Math.random() * REWARD_MUTATION_JITTER)
  nextRewardMutationTick = baselineState.tickCount + REWARD_MUTATION_INTERVAL + jitterDelay
  console.log(`[Baseline] Reward profile nudged: ${adjustments.join(', ')}`)
}

function getLineagePrestige(lineage) {
  const stats = lineageStats.get(lineage)
  return stats?.prestige ?? 0
}

async function ensureCrossoverBrain(slot) {
  while (crossoverBrains.length <= slot) {
    crossoverBrains.push(null)
  }
  if (crossoverBrains[slot]) {
    return crossoverBrains[slot]
  }
  try {
    const brain = await createBrain(OBS_SIZE, ACTIONS.length)
    crossoverBrains[slot] = brain
    return brain
  } catch (err) {
    if (isRemoteBrainUnavailableError(err)) {
      throw err
    }
    console.error('[Baseline] Failed to provision crossover brain:', err)
    return null
  }
}

async function performPopulationCrossover(sortedContexts) {
  if (!baselineBrain) return
  if (!Array.isArray(sortedContexts) || !sortedContexts.length) return

  const midIndex = Math.max(1, Math.floor(sortedContexts.length / 2))
  const peers = sortedContexts.slice(midIndex, midIndex + 2)
  const clones = []

  for (let i = 0; i < peers.length; i++) {
    try {
      const clone = await ensureCrossoverBrain(i)
      if (!clone) continue
      await copyWeights(clone, baselineBrain)
      const targetReward = peers[i]?.generationReward ?? 0
      const topReward = sortedContexts[i]?.generationReward ?? sortedContexts[0]?.generationReward ?? 0
      const gap = Math.max(0.1, Math.abs(topReward - targetReward))
      await mutateWeights(clone, Math.min(0.1, 0.01 + gap * 0.005))
      clones.push(clone)
    } catch (err) {
      if (isRemoteBrainUnavailableError(err)) {
        throw err
      }
      console.error('[Baseline] Failed preparing crossover clone:', err)
    }
  }

  if (!clones.length) {
    return
  }

  try {
    await averageWeights(baselineBrain, clones)
    await mutateWeights(baselineBrain, 0.015)
    markBaselineWeightsHealthy('population-crossover')
    console.log(`[Baseline] Applied population crossover with ${clones.length} companion models.`)
  } catch (err) {
    if (isRemoteBrainUnavailableError(err)) {
      throw err
    }
    console.error('[Baseline] Population crossover failed:', err)
  }
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
  const timeOfDay = Number(bot.time?.timeOfDay ?? bot.time?.day ?? bot.time?.age ?? 0)
  const normalizedTime = Number.isFinite(timeOfDay) ? ((timeOfDay % 24000) / 24000) : 0
  const isDay = normalizedTime >= 0.25 && normalizedTime <= 0.75 ? 1 : 0
  const weatherState = bot.world?.weather ?? (bot.isRaining ? 'rain' : 'clear')
  const raining = bot.isRaining || weatherState === 'rain' || weatherState === 'thunder' ? 1 : 0
  const thundering = bot.isThundering || weatherState === 'thunder' ? 1 : 0
  obs[22] = normalizedTime
  obs[23] = isDay
  obs[24] = raining ? 1 : 0
  obs[25] = thundering ? 1 : 0

  trackEnvironmentAwareness(context)
  trackNovelty(context, obs)

  const uniqueStateRatio = Math.min(1, context.visitedStates.size / NOVELTY_TARGET)
  const uniqueBlocksRatio = Math.min(1, context.visitedBlocks.size / 200)
  const uniqueBiomesRatio = Math.min(1, context.visitedBiomes.size / 32)
  obs[26] = uniqueStateRatio
  obs[27] = uniqueBlocksRatio
  obs[28] = uniqueBiomesRatio

  obs[29] = Math.min(1, context.resources.wood / 64)
  obs[30] = Math.min(1, context.resources.stone / 128)
  obs[31] = Math.min(1, context.resources.ore / 64)
  obs[32] = Math.min(1, context.resources.crafted / 32)
  obs[33] = Math.min(1, context.resourceLedger.contributed / 128)
  obs[34] = Math.min(1, context.resourceLedger.withdrawn / 128)
  obs[35] = Math.min(1, context.behaviorEntropy)
  obs[36] = Math.min(1, context.currentChainScore)
  obs[37] = context.stagnation.active ? 1 : 0
  const emotion = ensureEmotionVector(context)
  obs[38] = emotion[0] ?? MORALE_BASELINE
  obs[39] = emotion[1] ?? 0
  obs[40] = emotion[2] ?? 0.5
  obs[41] = context.mode === 'feral' ? 1 : 0
  const lineagePrestige = getLineagePrestige(context.lineage)
  obs[42] = Math.max(0, Math.min(1, lineagePrestige))
  const diversityRatio = Math.min(1, GLOBAL_RESOURCE_POOL.diversity.size / RESOURCE_TYPES.length)
  obs[43] = diversityRatio

  const tickCount = Number.isFinite(context.tickCount) ? context.tickCount : 0
  obs[44] = normalizePositive(tickCount, CONTEXT_TICK_NORMALIZER)
  const trainingSteps = Number.isFinite(context.trainingSteps) ? context.trainingSteps : 0
  obs[45] = normalizePositive(trainingSteps, CONTEXT_TRAINING_NORMALIZER)
  const cumulativeReward = Number.isFinite(context.cumulativeReward) ? context.cumulativeReward : 0
  obs[46] = normalizeMagnitude(cumulativeReward, CONTEXT_REWARD_NORMALIZER)
  const noveltyTotal = Number.isFinite(context.noveltyCount) ? context.noveltyCount : 0
  obs[47] = normalizePositive(noveltyTotal, NOVELTY_COUNT_NORMALIZER)

  const feralFury = Number.isFinite(context.feralFury) ? Math.max(0, context.feralFury) : 0
  obs[48] = normalizePositive(feralFury, FERAL_FURY_NORMALIZER)
  const cooperationScore = Number.isFinite(context.cooperationScore) ? context.cooperationScore : 0
  obs[49] = normalizeMagnitude(cooperationScore, COOPERATION_CLAMP)
  const epsilon = Number.isFinite(context.epsilon) ? Math.max(0, context.epsilon) : EPSILON_START
  obs[50] = normalizePositive(epsilon, EPSILON_NORMALIZER)
  const epsilonBoost = Number.isFinite(context.epsilonBoost) ? Math.abs(context.epsilonBoost) : 0
  obs[51] = normalizePositive(epsilonBoost, EPSILON_BOOST_NORMALIZER)
  const morale = context.morale ?? { successStreak: 0, failureStreak: 0, frustration: 0, sharpness: 0.5 }
  obs[52] = normalizePositive(Number(morale.successStreak) || 0, STREAK_NORMALIZER)
  obs[53] = normalizePositive(Number(morale.failureStreak) || 0, STREAK_NORMALIZER)
  obs[54] = normalizeMagnitude(Number(morale.frustration) || 0, MORALE_FRUSTRATION_CLAMP)
  obs[55] = normalizeMagnitude(Number(morale.sharpness) || 0, MORALE_SHARPNESS_CLAMP)

  const comm = ensureCommunicationState(context)
  const commSummary = comm?.summary
  if (commSummary) {
    for (let i = 0; i < COMMUNICATION_TYPE_COUNT; i++) {
      const allyValue = Number.isFinite(commSummary.allies?.[i]) ? commSummary.allies[i] : 0
      obs[56 + i] = normalizePositive(allyValue, COMMUNICATION_INTENSITY_CLAMP)
    }
    for (let i = 0; i < COMMUNICATION_TYPE_COUNT; i++) {
      const otherValue = Number.isFinite(commSummary.others?.[i]) ? commSummary.others[i] : 0
      obs[56 + COMMUNICATION_TYPE_COUNT + i] = normalizePositive(otherValue, COMMUNICATION_INTENSITY_CLAMP)
    }
    const now = Date.now()
    const lastSentAgo = comm?.lastSentAt ? Math.max(0, now - comm.lastSentAt) : Number.POSITIVE_INFINITY
    const lastHeardAgo = comm?.lastHeardAt ? Math.max(0, now - comm.lastHeardAt) : Number.POSITIVE_INFINITY
    const sentFreshness = 1 - normalizePositive(lastSentAgo, COMMUNICATION_TIME_NORMALIZER)
    const heardFreshness = 1 - normalizePositive(lastHeardAgo, COMMUNICATION_TIME_NORMALIZER)
    obs[56 + COMMUNICATION_TYPE_COUNT * 2] = Math.max(0, Math.min(1, sentFreshness))
    obs[57 + COMMUNICATION_TYPE_COUNT * 2] = Math.max(0, Math.min(1, heardFreshness))
    const direction = commSummary.direction ?? { x: 0, z: 0 }
    obs[58 + COMMUNICATION_TYPE_COUNT * 2] = normalizeMagnitude(
      Number(direction.x) || 0,
      COMMUNICATION_DIRECTION_NORMALIZER
    )
    obs[59 + COMMUNICATION_TYPE_COUNT * 2] = normalizeMagnitude(
      Number(direction.z) || 0,
      COMMUNICATION_DIRECTION_NORMALIZER
    )
  } else {
    for (let i = 0; i < COMMUNICATION_TYPE_COUNT * 2 + 4; i++) {
      obs[56 + i] = 0
    }
  }

  populateMemoryObservation(context, obs, MEMORY_OBS_START)

  return sanitizeVector(obs)
}

async function equipBestTool(context, preferredKeywords = []) {
  const items = context.bot.inventory?.items?.() ?? []
  for (const keyword of preferredKeywords) {
    const tool = items.find(item => item?.name?.includes(keyword))
    if (tool) {
      if (context.bot.heldItem?.type === tool.type) {
        return true
      }
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

function ensureMaintenanceState(context) {
  if (!context.maintenance) {
    context.maintenance = {
      nextAutoEatAt: 0,
      nextArmorCheckAt: 0,
      eating: false,
      lastAutoEatSuccess: 0,
      hungerCrisis: false,
      lastArmorSignature: null,
      lastArmorUpdate: 0
    }
  }
  return context.maintenance
}

function getItemName(context, item) {
  if (!item) return null
  if (item.name) return item.name
  if (typeof item.type === 'number') {
    return resolveRegistryItemName(context, item.type)
  }
  return null
}

function getItemInfo(context, item) {
  if (!context?.registry || !item) return null
  const name = getItemName(context, item)
  if (name && context.registry.itemsByName?.[name]) {
    return context.registry.itemsByName[name]
  }
  if (typeof item.type === 'number') {
    return context.registry.items?.[item.type] ?? null
  }
  return null
}

function getFoodPoints(context, item) {
  if (!item) return 0
  if (typeof item.foodPoints === 'number') return item.foodPoints
  const info = getItemInfo(context, item)
  if (info && typeof info.foodPoints === 'number') {
    return info.foodPoints
  }
  if (info && typeof info.foodPoints === 'object' && typeof info.foodPoints.default === 'number') {
    return info.foodPoints.default
  }
  return 0
}

function isCookedFood(name = '') {
  if (!name) return false
  return COOKED_FOOD_KEYWORDS.some(keyword => name.includes(keyword))
}

function isAvoidFood(name = '') {
  if (!name) return false
  return AVOID_FOOD_KEYWORDS.some(keyword => name.includes(keyword))
}

function isEdible(context, item) {
  if (!item) return false
  const name = getItemName(context, item)
  const info = getItemInfo(context, item)
  if (info?.edible) return true
  if (typeof item.foodPoints === 'number') return item.foodPoints > 0
  if (info && typeof info.foodPoints === 'number') return info.foodPoints > 0
  if (name && name.includes('bread')) return true
  if (name && name.includes('apple')) return true
  return false
}

function scoreFoodItem(context, item) {
  const name = (getItemName(context, item) ?? '').toLowerCase()
  const foodPoints = getFoodPoints(context, item)
  let score = foodPoints * 10
  if (isCookedFood(name)) score += 45
  if (name.includes('stew') || name.includes('soup')) score += 25
  if (name.includes('bread') || name.includes('pie') || name.includes('cake')) score += 15
  if (name.includes('raw')) score -= 18
  if (isAvoidFood(name)) score -= 35
  if (name.includes('golden_apple')) score += 60
  if (foodPoints === 0 && name.includes('honey')) score += 15
  return score
}

function armorSlotFromName(name = '') {
  if (!name) return null
  for (const { slot, keywords } of ARMOR_SLOT_PATTERNS) {
    if (keywords.some(keyword => name.includes(keyword))) {
      return slot
    }
  }
  return null
}

function scoreArmorItem(item) {
  if (!item) return 0
  const name = (item.name ?? '').toLowerCase()
  const slot = armorSlotFromName(name)
  if (!slot) return 0
  let score = ARMOR_SLOT_BONUS[slot] ?? 10
  for (const { keyword, score: tierScore } of ARMOR_TIER_WEIGHTS) {
    if (name.includes(keyword)) {
      score = Math.max(score, tierScore + (ARMOR_SLOT_BONUS[slot] ?? 0))
    }
  }
  if (name.includes('elytra')) {
    score += 40
  }
  if (name.includes('leather')) {
    score -= 10
  }
  if (name.includes('gold')) {
    score -= 5
  }
  return score
}

function getEquippedArmor(context, slot) {
  if (!context?.bot?.getEquipmentDestSlot) return null
  const slotId = context.bot.getEquipmentDestSlot(slot)
  if (typeof slotId !== 'number') return null
  return context.bot.inventory?.slots?.[slotId] ?? null
}

async function maybeAutoEquipArmor(context) {
  const maint = ensureMaintenanceState(context)
  const now = Date.now()
  if (now < maint.nextArmorCheckAt) {
    return false
  }
  maint.nextArmorCheckAt = now + ARMOR_CHECK_INTERVAL_MS

  const { bot } = context
  if (!bot?.inventory?.items) return false
  const items = bot.inventory.items() ?? []
  const slotBest = new Map()
  for (const item of items) {
    const name = getItemName(context, item)
    const slot = armorSlotFromName(name?.toLowerCase?.() ?? name ?? '')
    if (!slot) continue
    const score = scoreArmorItem(item)
    const current = slotBest.get(slot)
    if (!current || score > current.score) {
      slotBest.set(slot, { item, score })
    }
  }

  let equipped = false
  for (const [slot, candidate] of slotBest.entries()) {
    const current = getEquippedArmor(context, slot)
    const currentScore = scoreArmorItem(current)
    if (!current || candidate.score > currentScore + 0.5) {
      try {
        await bot.equip(candidate.item, slot)
        equipped = true
      } catch (err) {
        console.warn(`[${label(context)}] Failed to auto-equip ${candidate.item?.name ?? 'armor'}:`, err?.message ?? err)
      }
    }
  }

  if (equipped) {
    maint.lastArmorSignature = generateArmorSignature(context)
    maint.lastArmorUpdate = Date.now()
  }

  return equipped
}

function generateArmorSignature(context) {
  if (!context?.bot?.getEquipmentDestSlot) return null
  const slots = ['head', 'torso', 'legs', 'feet']
  const pieces = []
  for (const slot of slots) {
    const item = getEquippedArmor(context, slot)
    pieces.push(item?.name ?? 'none')
  }
  return pieces.join('|')
}

function isHungry(context) {
  const hunger = Number(context?.bot?.food ?? 20)
  return hunger < HUNGER_EAT_THRESHOLD
}

function isCriticalHunger(context) {
  const hunger = Number(context?.bot?.food ?? 20)
  return hunger <= HUNGER_CRITICAL_THRESHOLD
}

async function maybeAutoEat(context) {
  const maint = ensureMaintenanceState(context)
  const now = Date.now()
  if (now < maint.nextAutoEatAt) {
    return false
  }
  if (!context?.bot?.inventory?.items) {
    maint.nextAutoEatAt = now + AUTO_EAT_CHECK_INTERVAL_MS
    return false
  }
  const hunger = Number(context.bot.food ?? 20)
  const interval =
    hunger <= HUNGER_CRITICAL_THRESHOLD
      ? Math.max(800, Math.floor(AUTO_EAT_CHECK_INTERVAL_MS / 2))
      : AUTO_EAT_CHECK_INTERVAL_MS
  maint.nextAutoEatAt = now + interval
  if (hunger >= HUNGER_EAT_THRESHOLD) {
    maint.hungerCrisis = false
    return false
  }

  const items = context.bot.inventory.items() ?? []
  const edible = items.filter(item => isEdible(context, item))
  if (!edible.length) {
    maint.hungerCrisis = hunger <= HUNGER_HUNT_THRESHOLD
    return false
  }

  edible.sort((a, b) => scoreFoodItem(context, b) - scoreFoodItem(context, a))
  const best = edible[0]
  if (!best) {
    maint.hungerCrisis = hunger <= HUNGER_HUNT_THRESHOLD
    return false
  }

  try {
    maint.eating = true
    if (!context.bot.heldItem || context.bot.heldItem.type !== best.type) {
      await context.bot.equip(best, 'hand')
    }
    context.bot.clearControlStates()
    await context.bot.consume()
    await sleep(200)
    maint.lastAutoEatSuccess = Date.now()
    maint.hungerCrisis = false
    addBlockReward(
      context,
      hunger <= HUNGER_CRITICAL_THRESHOLD ? 0.25 : 0.12,
      REWARD_SIGN.POSITIVE,
      'auto-eat'
    )
    console.log(`[${label(context)}] Auto-ate ${best.name ?? 'food'} to restore hunger.`)
    return true
  } catch (err) {
    maint.hungerCrisis = hunger <= HUNGER_HUNT_THRESHOLD
    console.warn(`[${label(context)}] Auto-eat failed:`, err?.message ?? err)
  } finally {
    maint.eating = false
  }

  return false
}

function isPassiveAnimal(entity) {
  if (!entity) return false
  const kind = (entity.kind ?? entity.type ?? '').toLowerCase()
  if (kind.includes('passive')) return true
  const name = (entity.name ?? entity.displayName ?? '').toLowerCase()
  if (!name) return false
  return PASSIVE_ANIMAL_KEYWORDS.some(keyword => name.includes(keyword))
}

function isAttackableEntity(bot, entity, { maxDistance = 4.5 } = {}) {
  if (!bot || !entity) return false
  if (entity === bot.entity) return false
  if (bot.entity?.uuid && entity.uuid && bot.entity.uuid === entity.uuid) return false
  if (entity.isValid === false) return false
  if (!bot.entity?.position || !entity.position) return false

  let distance = Infinity
  try {
    distance = bot.entity.position.distanceTo(entity.position)
  } catch (err) {
    distance = Infinity
  }

  if (!Number.isFinite(distance) || distance > maxDistance) {
    return false
  }

  if (typeof entity.health === 'number' && entity.health <= 0) {
    return false
  }

  const type = (entity.type ?? '').toLowerCase()
  const kind = (entity.kind ?? '').toLowerCase()
  const name = (entity.name ?? entity.displayName ?? '').toLowerCase()

  const attackableType =
    type === 'mob' ||
    type === 'player' ||
    kind.includes('mob') ||
    kind.includes('hostile') ||
    kind.includes('neutral') ||
    kind.includes('passive') ||
    isPassiveAnimal(entity)

  if (!attackableType && !name) {
    return false
  }

  if (!attackableType) {
    const forbiddenKeywords = ['item', 'projectile', 'xp_orb', 'boat', 'minecart']
    if (forbiddenKeywords.some(keyword => name.includes(keyword) || type.includes(keyword))) {
      return false
    }
  }

  return true
}

function findAttackTarget(context, options = {}) {
  const bot = context?.bot
  if (!bot?.entity) return null
  const candidate = bot.nearestEntity(entity => isAttackableEntity(bot, entity, options))
  if (!candidate) return null
  return isAttackableEntity(bot, candidate, options) ? candidate : null
}

function isAnimalFoodItem(name = '') {
  const normalized = name.toLowerCase()
  return (
    normalized.includes('beef') ||
    normalized.includes('pork') ||
    normalized.includes('mutton') ||
    normalized.includes('chicken') ||
    normalized.includes('rabbit') ||
    normalized.includes('cod') ||
    normalized.includes('salmon') ||
    normalized.includes('steak') ||
    normalized.includes('bacon')
  )
}

async function runMaintenanceRoutines(context) {
  if (!context?.bot) return
  const maint = ensureMaintenanceState(context)
  maint.hungerCrisis = isHungry(context)

  let ate = false
  try {
    ate = await maybeAutoEat(context)
  } catch (err) {
    console.warn(`[${label(context)}] Auto-eat routine failed:`, err?.message ?? err)
  }

  try {
    await maybeAutoEquipArmor(context)
  } catch (err) {
    console.warn(`[${label(context)}] Auto-armor routine failed:`, err?.message ?? err)
  }

  return ate
}

function getToolTypeFromName(name = '') {
  if (!name) return null
  if (name.includes('pickaxe')) return 'pickaxe'
  if (name.includes('axe')) return 'axe'
  if (name.includes('shovel') || name.includes('spade')) return 'shovel'
  if (name.includes('hoe')) return 'hoe'
  if (name.includes('shears')) return 'shears'
  return null
}

function getToolTierScore(name = '') {
  let score = 0
  for (const { keyword, score: tierScore } of TOOL_TIER_WEIGHTS) {
    if (name.includes(keyword)) {
      score = Math.max(score, tierScore)
    }
  }
  return score
}

function resolveRegistryItemName(context, id) {
  if (!context?.registry || typeof id !== 'number' || Number.isNaN(id)) return null
  const { registry } = context
  const direct = registry.items?.[id]
  if (direct?.name) return direct.name
  if (Array.isArray(registry.items)) {
    const match = registry.items.find(item => item?.id === id)
    if (match?.name) return match.name
  }
  if (registry.itemsByName) {
    for (const [name, info] of Object.entries(registry.itemsByName)) {
      if (info?.id === id) return name
    }
  }
  return null
}

function determinePreferredToolTypes(context, block) {
  const order = []
  const addType = type => {
    if (type && !order.includes(type)) {
      order.push(type)
    }
  }

  if (block?.harvestTools && typeof block.harvestTools === 'object') {
    for (const key of Object.keys(block.harvestTools)) {
      const id = Number.parseInt(key, 10)
      if (Number.isNaN(id)) continue
      const name = resolveRegistryItemName(context, id)
      const toolType = getToolTypeFromName(name ?? '')
      addType(toolType)
    }
  }

  if (order.length === 0) {
    const material = block?.material ?? ''
    const name = block?.name ?? ''
    const addForTokens = (tokens, type) => {
      if (tokens.some(token => material.includes(token) || name.includes(token))) {
        addType(type)
      }
    }

    addForTokens(['stone', 'rock', 'ore', 'deepslate', 'nether', 'metal', 'anvil', 'obsidian'], 'pickaxe')
    addForTokens(['log', 'wood', 'stem', 'hyphae', 'plank', 'mushroom', 'pumpkin'], 'axe')
    addForTokens(['dirt', 'grass', 'sand', 'gravel', 'clay', 'snow', 'soul', 'mud', 'powder'], 'shovel')
    if (name.includes('leaves') || material.includes('leaf')) {
      addType('hoe')
    }
  }

  if (order.length === 0) {
    for (const fallback of DEFAULT_MINING_TOOL_PREFERENCES) {
      addType(fallback)
    }
  }

  return order
}

function scoreToolForBlock(context, item, block, preferredTypes) {
  if (!item?.name) return -Infinity
  const name = item.name
  const toolType = getToolTypeFromName(name)
  const harvestTools = block?.harvestTools
  const hasHarvestHints = harvestTools && Object.keys(harvestTools).length > 0

  let score = 0
  if (hasHarvestHints) {
    if (harvestTools[item.type]) {
      score += 600
    } else {
      score -= 60
    }
  }

  if (toolType && preferredTypes.includes(toolType)) {
    score += 180
  }

  if (preferredTypes.some(type => name.includes(type))) {
    score += 120
  }

  score += TOOL_TYPE_WEIGHTS[toolType] ?? 0
  score += getToolTierScore(name)

  return score
}

async function equipOptimalMiningTool(context, block) {
  const preferredTypes = determinePreferredToolTypes(context, block)
  const items = context.bot.inventory?.items?.() ?? []
  let best = null

  for (const item of items) {
    const score = scoreToolForBlock(context, item, block, preferredTypes)
    if (!best || score > best.score) {
      best = { item, score }
    }
  }

  if (best && best.score > 0) {
    if (context.bot.heldItem?.type === best.item.type) {
      return true
    }
    try {
      await context.bot.equip(best.item, 'hand')
      return true
    } catch (err) {
      console.warn(`[${label(context)}] Failed to equip ${best.item.name}:`, err?.message ?? err)
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
    addBlockReward(context, config.reward ?? 0.4, REWARD_SIGN.POSITIVE, 'craft-success')
    context.resources.crafted += config.amount ?? 1
    GLOBAL_RESOURCE_POOL.crafted += config.amount ?? 1
    noteResourceDiversity('crafted')
    registerWithdrawal(context, config.amount ?? 1)
    updateCooperationScore(context)
    if (context.mode === 'feral') {
      context.feralFury = Math.max(0, (context.feralFury ?? 0) - 0.3)
    }
    console.log(`[${label(context)}] Crafted ${config.item}`)
  } else {
    addBlockReward(context, 0.03, REWARD_SIGN.NEGATIVE, 'craft-failure')
  }
}

async function performMining(context, { forward = false, strafe = 0 } = {}) {
  const { bot } = context
  const target = bot.blockAtCursor(5)
  if (!target) {
    addBlockReward(context, 0.02, REWARD_SIGN.NEGATIVE, 'mining-no-target')
    return
  }

  const movementVector = {
    forward: forward ? 1 : 0,
    strafe: clampNumber(Number(strafe) || 0, -1, 1)
  }
  const releaseMovement = holdMovement(context, movementVector)

  let equipped = await equipOptimalMiningTool(context, target)
  if (!equipped) {
    const preferences = determinePreferredToolTypes(context, target)
    equipped = await equipBestTool(context, preferences)
  }
  if (!equipped) {
    await equipBestTool(context, ['hand'])
  }

  try {
    await bot.dig(target)
    addBlockReward(context, 0.5, REWARD_SIGN.POSITIVE, 'mining-success')
    const resourceType = categorizeResource(target.name)
    if (resourceType && typeof context.resources[resourceType] === 'number') {
      context.resources[resourceType] += 1
      GLOBAL_RESOURCE_POOL[resourceType] += 1
      noteResourceDiversity(resourceType)
      registerContribution(context, 1)
      updateCooperationScore(context)
    }
    if (context.mode === 'feral') {
      context.feralFury = Math.min(5, (context.feralFury ?? 0) + 0.4)
    }
  } catch (err) {
    console.warn(`[${label(context)}] Mining failed:`, err?.message ?? err)
    addBlockReward(context, 0.05, REWARD_SIGN.NEGATIVE, 'mining-failure')
    if (context.mode === 'feral') {
      context.feralFury = Math.max(0, (context.feralFury ?? 0) - 0.2)
    }
  } finally {
    releaseMovement?.({ immediate: false })
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

  const movePulse = (vector, duration = 350, options = {}) =>
    smoothMovementPulse(context, vector, duration, options)

  const lookBy = async (deltaYaw = 0, deltaPitch = 0) => {
    await smoothLookBy(context, deltaYaw, deltaPitch)
  }

  const getTargetBlock = () => bot.blockAtCursor(5)

  try {
    switch (act) {
      case 'move_forward':
        await movePulse({ forward: 1 })
        break
      case 'move_backward':
        await movePulse({ forward: -1 })
        break
      case 'strafe_left':
        await movePulse({ strafe: -1 })
        break
      case 'strafe_right':
        await movePulse({ strafe: 1 })
        break
      case 'jump':
        await movePulse({}, 350, { jump: true })
        break
      case 'jump_forward':
        await movePulse({ forward: 1 }, 500, { jump: true })
        break
      case 'sprint_forward':
        await movePulse({ forward: 1 }, 500, { sprint: true })
        break
      case 'sneak_forward':
        await movePulse({ forward: 1 }, 500, { sneak: true })
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
      case 'mine':
        await performMining(context)
        break
      case 'mine_forward':
        await performMining(context, { forward: true })
        break
      case 'strafe_mine_left':
        await performMining(context, { strafe: -1 })
        break
      case 'strafe_mine_right':
        await performMining(context, { strafe: 1 })
        break
      case 'attack': {
        const entity = findAttackTarget(context)
        if (entity && isAttackableEntity(bot, entity)) {
          const hungerBefore = Number(bot.food ?? 20)
          try {
            await bot.attack(entity)
            addBlockReward(context, 0.3, REWARD_SIGN.POSITIVE, 'attack-success')
            if (
              hungerBefore <= HUNGER_HUNT_THRESHOLD &&
              isPassiveAnimal(entity)
            ) {
              addBlockReward(
                context,
                HUNGER_HUNT_ACTION_REWARD,
                REWARD_SIGN.POSITIVE,
                'attack-hunt-bonus'
              )
              console.log(`[${label(context)}] Rewarding hunt on ${entity.name ?? entity.displayName ?? 'mob'} while hungry.`)
            }
            if (context.mode === 'feral') {
              context.feralFury = Math.min(5, (context.feralFury ?? 0) + 0.6)
            }
          } catch (err) {
            console.warn(`[${label(context)}] Attack failed:`, err?.message ?? err)
            addBlockReward(context, 0.05, REWARD_SIGN.NEGATIVE, 'attack-failure')
            if (context.mode === 'feral') {
              context.feralFury = Math.max(0, (context.feralFury ?? 0) - 0.1)
            }
          }
        } else if (entity) {
          console.debug(
            `[${label(context)}] Skipping attack on invalid target ${entity.name ?? entity.displayName ?? entity.type ?? 'entity'}.`
          )
          addBlockReward(context, 0.02, REWARD_SIGN.NEGATIVE, 'attack-invalid-target')
        } else {
          addBlockReward(context, 0.01, REWARD_SIGN.NEGATIVE, 'attack-no-target')
          if (context.mode === 'feral') {
            context.feralFury = Math.max(0, (context.feralFury ?? 0) - 0.05)
          }
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
              addBlockReward(context, 0.25, REWARD_SIGN.POSITIVE, 'build-success')
              registerWithdrawal(context, 1)
              updateCooperationScore(context)
            } catch (err) {
              console.warn(`[${label(context)}] Build failed:`, err?.message ?? err)
              addBlockReward(context, 0.02, REWARD_SIGN.NEGATIVE, 'build-place-failure')
            }
          } else {
            addBlockReward(context, 0.02, REWARD_SIGN.NEGATIVE, 'build-no-block-equipped')
          }
          if (placePos) {
            // noop - placeholder for potential future heuristics
          }
        } else {
          addBlockReward(context, 0.02, REWARD_SIGN.NEGATIVE, 'build-no-target')
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
                addBlockReward(context, 0.2, REWARD_SIGN.POSITIVE, 'build-above-success')
                registerWithdrawal(context, 1)
                updateCooperationScore(context)
              } catch (err) {
                console.warn(`[${label(context)}] Build above failed:`, err?.message ?? err)
                addBlockReward(context, 0.02, REWARD_SIGN.NEGATIVE, 'build-above-failure')
              }
            }
          }
        } else {
          addBlockReward(context, 0.02, REWARD_SIGN.NEGATIVE, 'build-above-no-block')
        }
        break
      }
      case 'build_forward': {
        const target = getTargetBlock()
        if (target) {
          const success = await equipPlaceableBlock(context)
          if (success) {
            try {
              await bot.placeBlock(target, new Vec3(1, 0, 0))
              await movePulse({ forward: 1 }, 200)
              addBlockReward(context, 0.22, REWARD_SIGN.POSITIVE, 'build-forward-success')
              registerWithdrawal(context, 1)
              updateCooperationScore(context)
            } catch (err) {
              console.warn(`[${label(context)}] Build forward failed:`, err?.message ?? err)
              addBlockReward(context, 0.02, REWARD_SIGN.NEGATIVE, 'build-forward-failure')
            }
          }
        } else {
          addBlockReward(context, 0.02, REWARD_SIGN.NEGATIVE, 'build-forward-no-target')
        }
        break
      }
      case 'use_item': {
        try {
          bot.activateItem()
          await sleep(300)
          bot.deactivateItem()
          addBlockReward(context, 0.05, REWARD_SIGN.POSITIVE, 'use-item')
        } catch (err) {
          console.warn(`[${label(context)}] Use-item action failed:`, err?.message ?? err)
        }
        break
      }
      case 'signal_resource':
      case 'signal_danger':
      case 'signal_assist':
      case 'signal_gather':
      case 'signal_status': {
        const signalType = act.slice('signal_'.length)
        await performCommunicationAction(context, signalType)
        break
      }
      default:
        break
    }
  } finally {
    releaseMovement(context)
    bot.clearControlStates()
  }
}

function updateStagnation(context, reward) {
  const history = context.rewardHistory
  history.push(reward)
  if (history.length > STAGNATION_WINDOW) {
    history.shift()
  }

  if (history.length < 4) {
    context.stagnation.active = false
    return
  }

  const avg = history.reduce((sum, value) => sum + value, 0) / history.length
  const variance = history.reduce((sum, value) => {
    const diff = value - avg
    return sum + diff * diff
  }, 0) / history.length

  if (variance < STAGNATION_VARIANCE_THRESHOLD) {
    context.stagnation.streak += 1
    context.stagnation.active = true
  } else {
    context.stagnation.streak = 0
    context.stagnation.active = false
  }
}

function updateGenerationTrend(context) {
  if (!context?.rewardTrend) return

  const history = context.rewardTrend.generationHistory
  history.push(context.generationReward)
  if (history.length > DOWNTREND_GENERATION_WINDOW) {
    history.splice(0, history.length - DOWNTREND_GENERATION_WINDOW)
  }

  const previousFlag = context.rewardTrend.flaggedDownward ?? false

  if (history.length < 2) {
    context.rewardTrend.downwardStreak = 0
    context.rewardTrend.flaggedDownward = false
    context.rewardTrend.lastSlope = 0
    context.rewardTrend.lastWindow = history.slice()
    return
  }

  const first = history[0]
  const last = history[history.length - 1]
  const slope = (last - first) / (history.length - 1 || 1)
  const netDrop = first - last
  const trendingDown = slope <= -DOWNTREND_SLOPE_THRESHOLD && netDrop >= DOWNTREND_MARGIN

  if (trendingDown) {
    context.rewardTrend.downwardStreak = (context.rewardTrend.downwardStreak ?? 0) + 1
  } else {
    context.rewardTrend.downwardStreak = 0
  }

  context.rewardTrend.flaggedDownward = context.rewardTrend.downwardStreak >= DOWNTREND_STREAK_LIMIT
  context.rewardTrend.lastSlope = slope
  context.rewardTrend.lastWindow = history.slice()
  context.rewardTrend.lastNetDrop = netDrop
  context.rewardTrend.lastUpdatedGeneration = baselineState.generation

  if (!previousFlag && context.rewardTrend.flaggedDownward) {
    const snapshot = context.rewardTrend.lastWindow.map(value => value.toFixed(2)).join(' → ')
    console.log(
      `[${label(context)}] Reward downtrend detected (window ${snapshot || 'n/a'}, slope ${slope.toFixed(3)}).`
    )
  } else if (previousFlag && !context.rewardTrend.flaggedDownward) {
    const snapshot = context.rewardTrend.lastWindow.map(value => value.toFixed(2)).join(' → ')
    console.log(`[${label(context)}] Reward downtrend cleared (window ${snapshot || 'n/a'}).`)
  }
}

function shouldCullForDowntrend(context) {
  if (!context?.rewardTrend?.flaggedDownward) return false
  const age = (baselineState.generation ?? 0) - (context.birthGeneration ?? 0)
  if (age < DOWNTREND_MIN_AGE_GENERATIONS) {
    return false
  }
  return true
}

function computeReward(context, obs) {
  let reward = createRewardAccumulator()

  ensureTemporalMemoryState(context)
  context.damageDebt = limitPositive((context.damageDebt ?? 0) * DAMAGE_DEBT_DECAY, DAMAGE_MEMORY_CLAMP)
  context.recentDamage = limitPositive((context.recentDamage ?? 0) * DAMAGE_RECENT_DECAY, DAMAGE_MEMORY_CLAMP)

  const pos = { x: obs[0], y: obs[1], z: obs[2] }
  if (context.lastPos) {
    const dx = pos.x - context.lastPos.x
    const dy = pos.y - context.lastPos.y
    const dz = pos.z - context.lastPos.z
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz)
    const horizontal = Math.sqrt(dx * dx + dz * dz)
    reward = applyRewardComponent(
      reward,
      Math.min(dist * 0.1, 0.5),
      REWARD_SIGN.POSITIVE,
      context,
      'movement-distance'
    )
    reward = applyRewardComponent(
      reward,
      Math.min(horizontal * 0.05, 0.25),
      REWARD_SIGN.POSITIVE,
      context,
      'movement-horizontal'
    )
    reward = applyRewardComponent(
      reward,
      Math.min(Math.abs(dy) * 0.05, 0.15),
      REWARD_SIGN.POSITIVE,
      context,
      'movement-vertical'
    )
  }

  const health = obs[8]
  if (Number.isFinite(health)) {
    if (health < context.lastHealth) {
      const damage = Math.max(0, context.lastHealth - health)
      reward = applyRewardComponent(
        reward,
        Math.min(1.5, damage * 0.5),
        REWARD_SIGN.NEGATIVE,
        context,
        'damage-taken'
      )
      context.damageDebt = limitPositive(context.damageDebt + damage * DAMAGE_DEBT_WEIGHT, DAMAGE_MEMORY_CLAMP)
      context.recentDamage = limitPositive(context.recentDamage + damage, DAMAGE_MEMORY_CLAMP)
    } else if (health > context.lastHealth) {
      reward = applyRewardComponent(
        reward,
        Math.min(1, (health - context.lastHealth) * 0.5),
        REWARD_SIGN.POSITIVE,
        context,
        'health-regain'
      )
    }
    context.lastHealth = health
  }

  const food = obs[9]
  if (Number.isFinite(food)) {
    if (food > context.lastFood) {
      reward = applyRewardComponent(
        reward,
        Math.min(0.5, (food - context.lastFood) * 0.1),
        REWARD_SIGN.POSITIVE,
        context,
        'food-gain'
      )
    } else if (food < context.lastFood) {
      reward = applyRewardComponent(
        reward,
        Math.min(0.5, (context.lastFood - food) * 0.05),
        REWARD_SIGN.NEGATIVE,
        context,
        'food-loss'
      )
    }
    context.lastFood = food
  }

  const invTotal = obs[18]
  if (Number.isFinite(invTotal)) {
    const delta = invTotal - context.lastInvTotal
    if (delta !== 0) {
      reward = applyRewardComponent(
        reward,
        Math.sign(delta) * Math.min(Math.abs(delta) * 0.2, 1.5),
        REWARD_SIGN.EITHER,
        context,
        'inventory-delta'
      )
    }
    context.lastInvTotal = invTotal
  }

  const nearestDist = obs[16]
  if (Number.isFinite(nearestDist) && nearestDist > 0 && nearestDist < 3) {
    reward = applyRewardComponent(
      reward,
      (3 - nearestDist) * 0.05,
      REWARD_SIGN.NEGATIVE,
      context,
      'threat-proximity'
    )
  }

  if (context.lastAction != null) {
    if (context.prevAction === context.lastAction) {
      context.repetitionStreak += 1
    } else {
      context.repetitionStreak = 0
    }
    const fatiguePenalty = Math.min(0.6, context.repetitionStreak * 0.05)
    reward = applyRewardComponent(reward, fatiguePenalty, REWARD_SIGN.NEGATIVE, context, 'action-fatigue')
  }

  if (context.noveltyFlag) {
    reward = applyRewardComponent(
      reward,
      0.12 * rewardProfile.novelty,
      REWARD_SIGN.POSITIVE,
      context,
      'novelty'
    )
  }

  updateCooperationScore(context)
  const cooperation = Number.isFinite(context.cooperationScore) ? context.cooperationScore : 0
  const entropy = Number.isFinite(context.behaviorEntropy) ? context.behaviorEntropy : 0
  const chainScore = Number.isFinite(context.currentChainScore) ? context.currentChainScore : 0
  reward = applyRewardComponent(
    reward,
    Math.max(-0.3, Math.min(0.3, cooperation * 0.5)) * rewardProfile.cooperation,
    REWARD_SIGN.EITHER,
    context,
    'cooperation'
  )
  reward = applyRewardComponent(
    reward,
    (entropy - 0.5) * 0.1 * rewardProfile.entropy,
    REWARD_SIGN.EITHER,
    context,
    'entropy'
  )
  reward = applyRewardComponent(
    reward,
    Math.min(0.25, chainScore * 0.2) * rewardProfile.skill,
    REWARD_SIGN.POSITIVE,
    context,
    'skill-chain'
  )

  if (!Number.isFinite(context.blockReward)) {
    context.blockReward = 0
  }
  const consumedBlockReward = context.blockReward
  reward = applyRewardComponent(
    reward,
    consumedBlockReward * rewardProfile.resource,
    REWARD_SIGN.EITHER,
    context,
    'block-reward'
  )
  context.blockReward = 0

  const achievementBonus = context.achievementReward ?? 0
  if (achievementBonus !== 0) {
    reward = applyRewardComponent(reward, achievementBonus, REWARD_SIGN.POSITIVE, context, 'achievement')
    context.achievementReward = 0
  }

  if (context.damageDebt > 0) {
    reward = applyRewardComponent(
      reward,
      Math.min(2, context.damageDebt * DAMAGE_DEBT_PENALTY),
      REWARD_SIGN.NEGATIVE,
      context,
      'damage-debt'
    )
  }

  const pendingDeathPenalty = context.deathPenalty ?? 0
  if (pendingDeathPenalty > 0) {
    reward = applyRewardComponent(
      reward,
      pendingDeathPenalty,
      REWARD_SIGN.NEGATIVE,
      context,
      'death-penalty'
    )
    context.deathPenalty = 0
  }

  const lineageBonus = getLineagePrestige(context.lineage) * 0.1 * rewardProfile.lineage
  reward = applyRewardComponent(reward, lineageBonus, REWARD_SIGN.POSITIVE, context, 'lineage-prestige')

  if (context.mode === 'feral') {
    const fury = Number.isFinite(context.feralFury) ? context.feralFury : 0
    reward = applyRewardComponent(
      reward,
      Math.min(0.5, fury * 0.08) * rewardProfile.feral,
      REWARD_SIGN.POSITIVE,
      context,
      'feral-fury'
    )
    reward = applyRewardComponent(
      reward,
      Math.max(0, context.cooperationScore) * 0.2,
      REWARD_SIGN.NEGATIVE,
      context,
      'feral-cooperation-penalty'
    )
    context.feralFury = Math.max(0, fury * 0.92)
  } else {
    const fury = Number.isFinite(context.feralFury) ? context.feralFury : 0
    context.feralFury = Math.max(0, fury * 0.85)
  }

  reward = applyRewardComponent(reward, 0.02, REWARD_SIGN.NEGATIVE, context, 'tick-cost')

  reward = adjustMorale(context, reward)

  const summary = finalizeRewardAccumulator(reward)

  updateStagnation(context, summary.total)
  updateSkillChains(context, summary.total)

  context.lineagePrestige = getLineagePrestige(context.lineage)

  context.lastPos = { ...pos }
  return summary
}

async function tickLoop(context) {
  if (!globalRunning || !context.running || context.tickInFlight) return
  context.tickInFlight = true

  const tickStart = monotonicNow()
  let remoteUnavailable = false
  let remoteIssue = null
  try {
    if (!context.bot?.entity?.position) {
      console.warn(`[${label(context)}] Entity not ready, skipping tick.`)
      return
    }

    if (!context.registry && context.bot.registry) {
      context.registry = context.bot.registry
    }

    await runMaintenanceRoutines(context)

    let fallbackActive = Boolean(context.echoFallbackActive)
    const updateFallbackState = active => {
      const previous = Boolean(context.echoFallbackActive)
      if (active === previous) {
        return
      }
      context.echoFallbackActive = active
      if (active) {
        console.warn(
          `[${label(context)}] TensorFlow brain unavailable; switching to Echo fallback.`
        )
      } else {
        console.log(
          `[${label(context)}] TensorFlow brain reachable; exiting Echo fallback.`
        )
      }
    }

    const status = getRemoteBrainStatus()
    if (!status.connected) {
      remoteUnavailable = true
      remoteIssue = status
      if (isEchoFallbackEnabled()) {
        fallbackActive = true
        updateFallbackState(true)
      } else {
        updateFallbackState(false)
        return
      }
    } else if (fallbackActive) {
      fallbackActive = false
      updateFallbackState(false)
    } else {
      updateFallbackState(false)
    }

    if (context.weightsSuspect || context.pendingWeightRecovery) {
      await attemptWeightRecovery(context, 'tick')
      if (context.weightsSuspect) {
        if (!context.weightSkipNotified) {
          console.warn(
            `[${label(context)}] Brain weights suspect; deferring tick until recovery completes.`
          )
          context.weightSkipNotified = true
        }
        return
      }
    }

    let brain = null
    if (!fallbackActive) {
      try {
        brain = await ensureContextBrain(context)
      } catch (err) {
        if (isRemoteBrainUnavailableError(err) && isEchoFallbackEnabled()) {
          remoteUnavailable = true
          remoteIssue = err
          fallbackActive = true
          updateFallbackState(true)
        } else {
          throw err
        }
      }
    }
    if (!fallbackActive && !brain) {
      console.warn(`[${label(context)}] Brain not ready, skipping tick.`)
      return
    }

    updateCommunicationAwareness(context)
    const observation = gatherObservations(context)
    sanitizeVector(observation)
    if (!vectorHasFiniteValues(observation)) {
      console.warn(`[${label(context)}] Observation contained invalid values; skipping tick.`)
      return
    }
    const reward = computeReward(context, observation)

    updateTemporalMemory(context, reward.total)

    if (context.stagnation.active && context.stagnation.streak >= STAGNATION_WINDOW / 2) {
      context.epsilonBoost = Math.max(context.epsilonBoost, EPSILON_STAGNATION_BOOST)
    } else {
      context.epsilonBoost *= EPSILON_BOOST_DECAY
      if (context.epsilonBoost < 0.01) {
        context.epsilonBoost = 0
      }
    }

    const moraleState = context.morale ?? { value: MORALE_BASELINE, frustration: 0 }
    const moraleInfluence = (moraleState.value ?? MORALE_BASELINE) - MORALE_BASELINE
    const frustrationInfluence = moraleState.frustration ?? 0
    const epsilonBase = context.epsilon + context.epsilonBoost - moraleInfluence * 0.15 + frustrationInfluence * 0.1
    const effectiveEpsilon = Math.min(0.95, Math.max(EPSILON_MIN, epsilonBase))

    let trained = false
    if (context.lastObs && context.lastAction != null) {
      const lastObsValid = vectorHasFiniteValues(context.lastObs)
      if (lastObsValid) {
        let trainOutcome = null
        let trainUsedFallback = fallbackActive
        if (!fallbackActive) {
          try {
            trainOutcome = await trainBrainConcurrent(
              brain,
              context.lastObs,
              context.lastAction,
              reward.reward,
              reward.penalty,
              observation
            )
          } catch (err) {
            if (isRemoteBrainUnavailableError(err) && isEchoFallbackEnabled()) {
              remoteUnavailable = true
              remoteIssue = err
              fallbackActive = true
              trainUsedFallback = true
              updateFallbackState(true)
            } else {
              throw err
            }
          }
        }
        if (fallbackActive) {
          trainUsedFallback = true
          trainOutcome = await reportEchoLearning({
            observation: context.lastObs,
            nextObservation: observation,
            actionIndex: context.lastAction,
            actions: ACTIONS,
            reward: reward.total,
            botId: label(context)
          })
        }
        if (trainOutcome) {
          recordTrainingSanitization(trainOutcome.sanitization)
          recordDroppedGradients(trainOutcome.droppedGradients)
          recordClippedGradients(trainOutcome.clippedGradients)
          recordGradientNorm(trainOutcome.gradientNorm)
          trained = Boolean(trainOutcome.trained)
          if (!trainUsedFallback && trainOutcome.weightsOk === false) {
            const trainDetails = {
              ...(trainOutcome.sanitization ?? {}),
              trigger: 'train'
            }
            scheduleWeightRecovery(context, 'train-non-finite', trainDetails)
            console.warn(
              `[${label(context)}] Non-finite weights detected after training; deferring tick until recovery.`
            )
            return
          }
        }
      } else {
        console.warn(`[${label(context)}] Skipping training due to invalid previous observation values.`)
      }
    }

    let actionResult = null
    let actionUsedFallback = fallbackActive
    if (!fallbackActive) {
      try {
        actionResult = await chooseActionConcurrent(brain, observation, effectiveEpsilon)
      } catch (err) {
        if (isRemoteBrainUnavailableError(err) && isEchoFallbackEnabled()) {
          remoteUnavailable = true
          remoteIssue = err
          fallbackActive = true
          actionUsedFallback = true
          updateFallbackState(true)
        } else {
          throw err
        }
      }
    }
    if (fallbackActive) {
      actionResult = await chooseEchoAction({
        observation,
        epsilon: effectiveEpsilon,
        actions: ACTIONS,
        botId: label(context)
      })
      actionUsedFallback = true
    }
    if (!actionResult) {
      throw new Error('Failed to select action for current tick')
    }
    recordActionSanitization(actionResult?.sanitization)
    const remotePolicyReplaced = Number.parseInt(
      actionResult?.sanitization?.remote?.policy?.replaced ?? 0,
      10
    )
    if (
      !actionUsedFallback &&
      (actionResult?.weightsOk === false ||
        (Number.isFinite(remotePolicyReplaced) && remotePolicyReplaced > 0))
    ) {
      const actionDetails = {
        observation: actionResult?.sanitization?.observation ?? {},
        remote: actionResult?.sanitization?.remote ?? {},
        trigger: 'act'
      }
      scheduleWeightRecovery(context, 'act-non-finite', actionDetails)
      console.warn(
        `[${label(context)}] Non-finite action distribution detected; deferring tick until recovery.`
      )
      return
    }

    const action = actionResult.action
    await executeAction(context, action)

    recordActionMemory(context, action)

    const actionLabel = ACTIONS[action] ?? String(action)
    updateBehaviorEntropy(context, actionLabel)

    context.prevAction = context.lastAction
    context.lastObs = observation
    context.lastAction = action
    context.tickCount += 1
    context.generationTicks += 1
    context.cumulativeReward += reward.total
    context.generationReward += reward.total
    baselineState.tickCount += 1
    baselineState.cumulativeReward += reward.total
    if (baselineState.tickCount >= nextRewardMutationTick) {
      mutateRewardProfile()
    }
    if (trained) {
      context.trainingSteps += 1
      baselineState.trainingSteps += 1
    }

    if (context.epsilon > EPSILON_MIN) {
      context.epsilon = Math.max(EPSILON_MIN, context.epsilon * EPSILON_DECAY)
    }

    if (context.epsilonBoost > 0) {
      context.epsilonBoost *= EPSILON_BOOST_DECAY
      if (context.epsilonBoost < 0.01) {
        context.epsilonBoost = 0
      }
    }

    console.log(
      `[${label(context)}] Tick done | Reward: ${reward.total.toFixed(3)} (R=${reward.reward.toFixed(3)} P=${reward.penalty.toFixed(3)}) | Eps: ${effectiveEpsilon.toFixed(3)} | Entropy: ${context.behaviorEntropy.toFixed(2)}`
    )
    maybeTriggerAutosave()
    await maybeCompleteGeneration(context)
  } catch (err) {
    if (isRemoteBrainUnavailableError(err)) {
      remoteUnavailable = true
      remoteIssue = err
    } else {
      console.error(`[${label(context)}] Tick error:`, err)
    }
  } finally {
    recordTickDuration(monotonicNow() - tickStart)
    maybeLogHealthSummary()
    context.tickInFlight = false

    if (remoteUnavailable) {
      const status = getRemoteBrainStatus()
      noteRemoteBrainOffline(context, status, remoteIssue)
    } else {
      noteRemoteBrainOnline(context)
      if (deferredBaselineSaveReason && !saveInFlight && isRemoteBrainConnected()) {
        const reason = deferredBaselineSaveReason
        deferredBaselineSaveReason = null
        scheduleBaselineSave(reason)
      }
    }

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

function deriveLineageDominancePlan(lineageCounts, total) {
  if (!lineageCounts || !lineageCounts.size || total <= 0) {
    return null
  }
  const entries = [...lineageCounts.entries()].sort((a, b) => (b[1] ?? 0) - (a[1] ?? 0))
  if (!entries.length) return null
  const [topName, topCountRaw] = entries[0]
  const topCount = Number(topCountRaw) || 0
  if (topCount <= 0) {
    return null
  }
  const share = topCount / total
  const nextShare = entries[1] ? (Number(entries[1][1]) || 0) / total : 0
  const gap = share - nextShare
  const stats = ensureLineageRecord(topName)
  if (
    share >= LINEAGE_DOMINANCE_SHARE_THRESHOLD &&
    gap >= LINEAGE_DOMINANCE_GAP_THRESHOLD
  ) {
    stats.dominanceStreak = (stats.dominanceStreak ?? 0) + 1
  } else {
    stats.dominanceStreak = 0
  }
  for (let i = 1; i < entries.length; i++) {
    const name = entries[i]?.[0]
    if (!name) continue
    const entryStats = ensureLineageRecord(name)
    if (entryStats !== stats) {
      entryStats.dominanceStreak = Math.max(0, entryStats.dominanceStreak ?? 0)
    }
  }
  const generation = baselineState.generation ?? 0
  if (
    stats.dominanceStreak >= LINEAGE_DOMINANCE_STREAK_THRESHOLD &&
    generation - (stats.lastDominanceMitigation ?? -Infinity) >= LINEAGE_DOMINANCE_COOLDOWN_GENERATIONS
  ) {
    return {
      lineage: topName,
      share,
      gap,
      count: topCount,
      stats,
      generation
    }
  }
  return null
}

async function synchronizeGeneration() {
  if (generationSyncInFlight) return
  if (!contexts.length) return
  if (!contexts.every(ctx => ctx.readyForSync)) return

  if (!isRemoteBrainConnected()) {
    if (isEchoFallbackEnabled()) {
      const status = getRemoteBrainStatus()
      console.warn(
        `[Baseline] Skipping generation sync while TensorFlow brain unavailable (${describeRemoteRetry(status)}).`
      )
    }
    return
  }

  generationSyncInFlight = true
  try {
    const sorted = [...contexts].sort((a, b) => b.generationReward - a.generationReward)
    const survivorCount = Math.max(
      MIN_BOTS,
      Math.min(GENERATION_SURVIVOR_COUNT, sorted.length)
    )
    const survivorList = sorted.slice(0, survivorCount)
    const survivorSet = new Set(survivorList)
    const topTwo = sorted.slice(0, 2)
    const rewardSnapshot = new Map()
    for (const ctx of sorted) {
      rewardSnapshot.set(ctx, ctx.generationReward ?? 0)
    }
    const lineageCounts = new Map()
    for (const ctx of contexts) {
      const lineageName = ctx?.lineage || LINEAGE_ROOT_NAME
      lineageCounts.set(lineageName, (lineageCounts.get(lineageName) ?? 0) + 1)
    }
    const dominancePlan = deriveLineageDominancePlan(lineageCounts, contexts.length)
    if (dominancePlan) {
      dominancePlan.stats.lastDominanceMitigation = dominancePlan.generation
      console.warn(
        `[Baseline] Lineage ${dominancePlan.lineage} controls ${(dominancePlan.share * 100).toFixed(
          1
        )}% of bots (gap ${(dominancePlan.gap * 100).toFixed(1)}%). Diversifying offspring with extra mutations.`
      )
    }
    await ensureBaselineReady()

    baselineState.generation += 1
    const leaderboard = topTwo.length
      ? topTwo.map(ctx => `${label(ctx)}=${ctx.generationReward.toFixed(2)}`).join(', ')
      : 'n/a'
    console.log(`[Baseline] Generation ${baselineState.generation} | Top rewards: ${leaderboard}`)

    const topReward = sorted[0]?.generationReward ?? -Infinity
    const averageReward = contexts.reduce((sum, ctx) => sum + ctx.generationReward, 0) / contexts.length

    const templateSources = []
    for (const candidate of sorted) {
      if (candidate?.brain?.id && !candidate.weightsSuspect) {
        templateSources.push(candidate.brain)
      }
      if (templateSources.length >= TEMPLATE_TOP_BRAINS) {
        break
      }
    }

    if (templateSources.length) {
      try {
        await averageWeights(baselineBrain, templateSources)
        markBaselineWeightsHealthy('template-average')
        console.log(
          `[Baseline] Updated template from top ${templateSources.length} brain${
            templateSources.length === 1 ? '' : 's'
          }.`
        )
      } catch (err) {
        if (isRemoteBrainUnavailableError(err)) {
          throw err
        }
        console.error('[Baseline] Failed to average top brains into template:', err)
      }
    } else {
      console.log('[Baseline] Skipped template averaging — no trained brains available.')
    }

    if (topReward > bestGenerationReward + 0.5) {
      bestGenerationReward = topReward
      stagnantGenerations = 0
    } else {
      stagnantGenerations += 1
    }

    if (stagnantGenerations >= STAGNATION_MUTATION_THRESHOLD) {
      console.log('[Baseline] Stagnation detected — triggering meta-mutation and epsilon reset.')
      try {
        await mutateWeights(baselineBrain, 0.05)
        markBaselineWeightsHealthy('stagnation-mutate')
        for (const ctx of contexts) {
          ctx.epsilon = Math.min(0.9, Math.max(ctx.epsilon, EPSILON_START))
          ctx.epsilonBoost = Math.max(ctx.epsilonBoost, EPSILON_STAGNATION_BOOST)
          ctx.stagnation.lastMutation = baselineState.generation
        }
      } catch (err) {
        if (isRemoteBrainUnavailableError(err)) {
          const status = getRemoteBrainStatus()
          console.warn(`[Baseline] Meta-mutation skipped: remote brain unavailable (${describeRemoteRetry(status)}).`)
        } else {
          console.error('[Baseline] Meta-mutation failed:', err)
        }
      }
      stagnantGenerations = 0
    }

    if (baselineState.generation % CROSSOVER_GENERATION_INTERVAL === 0) {
      await performPopulationCrossover(sorted)
    }

    let expansionCount = 0
    if (averageReward > 1 && contexts.length < MAX_BOTS) {
      expansionCount = 1
    }
    if (averageReward > 3 && contexts.length + expansionCount < MAX_BOTS) {
      expansionCount += 1
    }
    if (averageReward > 6 && contexts.length + expansionCount < MAX_BOTS) {
      expansionCount += 1
    }
    expansionCount = Math.min(expansionCount, MAX_BOTS - contexts.length)

    let contractionCount = 0
    if (averageReward < -0.5 && contexts.length > MIN_BOTS) {
      contractionCount = 1
    }
    if (averageReward < -2 && contexts.length - contractionCount > MIN_BOTS) {
      contractionCount += 1
    }
    contractionCount = Math.min(contractionCount, contexts.length - MIN_BOTS)

    const contractionCandidates = sorted.filter(ctx => !survivorSet.has(ctx))
    const retireList =
      contractionCount > 0 ? contractionCandidates.slice(-contractionCount) : []

    for (const ctx of contexts) {
      updateGenerationTrend(ctx)
    }

    const retireReasons = new Map()
    for (const retiree of retireList) {
      if (retiree) {
        retireReasons.set(retiree, 'dynamic-scaling')
      }
    }

    const availableDowntrendSlots = Math.max(0, contexts.length - retireReasons.size - MIN_BOTS)
    const downtrendRetirees = []
    if (availableDowntrendSlots > 0) {
      const downtrendCandidates = sorted
        .filter(
          ctx => !retireReasons.has(ctx) && !survivorSet.has(ctx) && shouldCullForDowntrend(ctx)
        )
        .sort((a, b) => {
          const ageOrder = (a.birthOrder ?? 0) - (b.birthOrder ?? 0)
          if (ageOrder !== 0) return ageOrder
          const slopeA = a.rewardTrend?.lastSlope ?? 0
          const slopeB = b.rewardTrend?.lastSlope ?? 0
          return slopeA - slopeB
        })
      const limit = Math.min(DOWNTREND_RETIRE_LIMIT, availableDowntrendSlots)
      for (const candidate of downtrendCandidates.slice(0, limit)) {
        retireReasons.set(candidate, 'reward-downtrend')
        downtrendRetirees.push(candidate)
      }
    }

    const lowRewardRetirees = []
    const availableLowRewardSlots = Math.max(
      0,
      Math.min(LOW_REWARD_RETIRE_LIMIT, contexts.length - retireReasons.size - MIN_BOTS)
    )
    if (availableLowRewardSlots > 0) {
      const eligible = sorted.filter(ctx => !retireReasons.has(ctx) && !survivorSet.has(ctx))
      const applyThreshold = Number.isFinite(LOW_REWARD_RETIRE_THRESHOLD)
      const thresholdCandidates = applyThreshold
        ? eligible.filter(ctx => (ctx.generationReward ?? 0) <= LOW_REWARD_RETIRE_THRESHOLD)
        : eligible
      const rankingPool = (thresholdCandidates.length ? thresholdCandidates : eligible).slice()
      rankingPool.sort((a, b) => {
        const genDelta = (a.generationReward ?? 0) - (b.generationReward ?? 0)
        if (genDelta !== 0) return genDelta
        const cumulativeDelta = (a.cumulativeReward ?? 0) - (b.cumulativeReward ?? 0)
        if (cumulativeDelta !== 0) return cumulativeDelta
        return (a.birthOrder ?? 0) - (b.birthOrder ?? 0)
      })
      for (const candidate of rankingPool.slice(0, availableLowRewardSlots)) {
        retireReasons.set(candidate, 'lowest-reward')
        lowRewardRetirees.push(candidate)
      }
    }

    const oldestRetirees = []
    const oldestSlots = Math.max(
      0,
      Math.min(OLDEST_RETIRE_PER_GENERATION, contexts.length - retireReasons.size - MIN_BOTS)
    )
    if (oldestSlots > 0) {
      const oldestPool = sorted
        .filter(ctx => !retireReasons.has(ctx) && !survivorSet.has(ctx))
        .sort((a, b) => {
          const genDelta = (a.birthGeneration ?? 0) - (b.birthGeneration ?? 0)
          if (genDelta !== 0) return genDelta
          const orderDelta = (a.birthOrder ?? 0) - (b.birthOrder ?? 0)
          if (orderDelta !== 0) return orderDelta
          return (a.birthTick ?? 0) - (b.birthTick ?? 0)
        })
      for (const candidate of oldestPool.slice(0, oldestSlots)) {
        retireReasons.set(candidate, 'oldest')
        oldestRetirees.push(candidate)
      }
    }

    for (const survivor of survivorSet) {
      retireReasons.delete(survivor)
    }

    const generationResetRetirees = []
    for (const candidate of sorted) {
      if (!survivorSet.has(candidate) && !retireReasons.has(candidate)) {
        retireReasons.set(candidate, 'generation-reset')
        generationResetRetirees.push(candidate)
      }
    }

    for (const ctx of contexts) {
      ctx.generationTicks = 0
      ctx.generationReward = 0
      ctx.readyForSync = false
    }

    for (const retiree of downtrendRetirees) {
      const trend = retiree.rewardTrend
      const slope = trend?.lastSlope ?? 0
      const history = trend?.lastWindow ?? []
      const snapshot = history.length ? history.map(value => value.toFixed(2)).join(' → ') : 'n/a'
      console.log(
        `[Baseline] Retiring ${label(retiree)} due to sustained reward downtrend (slope ${slope.toFixed(3)}, window ${snapshot}).`
      )
    }

    for (const retiree of lowRewardRetirees) {
      const genReward = retiree.generationReward ?? 0
      const totalReward = retiree.cumulativeReward ?? 0
      console.log(
        `[Baseline] Retiring ${label(retiree)} due to lowest generation reward (${genReward.toFixed(
          2
        )} gen, total ${totalReward.toFixed(2)}).`
      )
    }

    for (const retiree of oldestRetirees) {
      const ageGenerations = (baselineState.generation ?? 0) - (retiree.birthGeneration ?? 0)
      const ageTicks = (baselineState.tickCount ?? 0) - (retiree.birthTick ?? 0)
      console.log(
        `[Baseline] Retiring ${label(retiree)} as oldest member (age ${ageGenerations} gen, ${ageTicks} ticks).`
      )
    }

    for (const retiree of generationResetRetirees) {
      const rank = sorted.indexOf(retiree) + 1
      console.log(
        `[Baseline] Retiring ${label(retiree)} due to generation reset (rank #${rank}).`
      )
    }

    for (const [retiree, reason] of retireReasons.entries()) {
      await retireContext(retiree, reason)
    }

    const activeLineages = new Set()
    for (const ctx of contexts) {
      if (!ctx?.lineage) continue
      const stats = ensureLineageRecord(ctx.lineage)
      stats.survivalStreak = (stats.survivalStreak ?? 0) + 1
      stats.lastSeenGeneration = baselineState.generation
      stats.prestige = Math.min(0.6, (stats.prestige ?? 0) * 0.9 + 0.05)
      activeLineages.add(ctx.lineage)
    }
    for (const [name, stats] of lineageStats.entries()) {
      if (!activeLineages.has(name)) {
        stats.survivalStreak = Math.max(0, (stats.survivalStreak ?? 0) - 1)
        stats.prestige = Math.max(0, (stats.prestige ?? 0) * 0.85)
      }
    }
    for (const ctx of contexts) {
      ctx.lineagePrestige = getLineagePrestige(ctx.lineage)
    }

    const spawnRequests = Math.max(0, expansionCount + retireReasons.size)
    const availableSpawnSlots = Math.min(Math.max(0, MAX_BOTS - contexts.length), spawnRequests)
    if (availableSpawnSlots > 0) {
      const rewardCompare = (a, b) => (rewardSnapshot.get(b) ?? 0) - (rewardSnapshot.get(a) ?? 0)
      let parentPool = contexts
        .filter(ctx => survivorSet.has(ctx) && !ctx.weightsSuspect)
        .sort(rewardCompare)
      if (!parentPool.length) {
        parentPool = contexts.filter(ctx => survivorSet.has(ctx)).sort(rewardCompare)
      }
      let fallbackPool = survivorList.filter(ctx => !ctx.weightsSuspect)
      if (!fallbackPool.length) {
        fallbackPool = survivorList.slice()
      }
      if (!parentPool.length && fallbackPool.length) {
        parentPool = fallbackPool.slice()
      }
      if (!parentPool.length) {
        parentPool = survivorList.slice()
      }

      let spawnIndex = contexts.length
      const topBaselineReward = Number.isFinite(topReward) ? topReward : 0
      for (let i = 0; i < availableSpawnSlots; i++) {
        const parentCandidate = parentPool[i % parentPool.length] ?? sorted[0] ?? null
        const partnerCandidate =
          parentPool[(i + 1) % parentPool.length] ?? survivorList[(i + 1) % survivorList.length] ?? null
        const parentReward = rewardSnapshot.get(parentCandidate) ?? 0
        const rewardGap = Math.max(0, topBaselineReward - parentReward)
        const baseMutation = NEW_BRAIN_MUTATION_STDDEV + rewardGap * MUTATION_REWARD_FACTOR
        let mutationStddev = deriveMutationStddev(baseMutation, { jitter: true })
        if (!Number.isFinite(mutationStddev) || mutationStddev <= 0) {
          mutationStddev = deriveMutationStddev(null, { jitter: true }) ?? NEW_BRAIN_MUTATION_STDDEV
        }
        const partnerContext =
          partnerCandidate && partnerCandidate !== parentCandidate ? partnerCandidate : null
        let effectiveMutation = mutationStddev
        const extraMutations = []
        let lineageOverride = null
        if (dominancePlan && parentCandidate?.lineage === dominancePlan.lineage) {
          const variantBase = deriveLineageVariantBase(parentCandidate.lineage, dominancePlan.stats)
          lineageOverride = variantBase
          const boosted = deriveMutationStddev(
            effectiveMutation * (1 + LINEAGE_DOMINANCE_MUTATION_MULTIPLIER),
            { jitter: true }
          )
          if (Number.isFinite(boosted) && boosted > 0) {
            effectiveMutation = Math.min(NEW_BRAIN_MUTATION_MAX, boosted)
          }
          for (let extra = 0; extra < LINEAGE_DOMINANCE_EXTRA_MUTATIONS; extra++) {
            const scaled = effectiveMutation * (1 + extra * 0.15)
            const derived = deriveMutationStddev(scaled, { jitter: true })
            if (Number.isFinite(derived) && derived > 0) {
              extraMutations.push(derived)
            }
          }
          dominancePlan.stats.instability = Math.max(0, (dominancePlan.stats.instability ?? 0) * 0.5)
          console.log(
            `[Baseline] Diversifying ${parentCandidate ? label(parentCandidate) : 'unknown'} → ${variantBase} with ${extraMutations.length} extra mutation${
              extraMutations.length === 1 ? '' : 's'
            }.`
          )
        }
        const spawnOptions = {
          parent: parentCandidate,
          partner: partnerContext,
          mutationStddev: effectiveMutation
        }
        if (lineageOverride) {
          spawnOptions.lineageBase = lineageOverride
        }
        if (extraMutations.length) {
          spawnOptions.extraMutations = extraMutations
        }
        createContext(spawnIndex++, spawnOptions)
      }
    }

    for (const stats of lineageStats.values()) {
      stats.instability = Math.max(0, (stats.instability ?? 0) * 0.85)
    }

    scheduleBaselineSave('generation')
  } catch (err) {
    if (isRemoteBrainUnavailableError(err)) {
      const status = getRemoteBrainStatus()
      console.warn(`[Baseline] Skipping generation sync: remote brain unavailable (${describeRemoteRetry(status)}).`)
    } else {
      console.error('[Baseline] Failed to synchronize generation:', err)
    }
  } finally {
    generationSyncInFlight = false
  }
}

function applyDeathPenalty(context, source = 'unknown') {
  if (!context) return
  const now = Date.now()
  if (context.lastDeathAt && now - context.lastDeathAt < 1000) {
    return
  }
  context.lastDeathAt = now
  const penalty = Math.max(5, DEATH_REWARD_PENALTY)
  context.deathPenalty = (context.deathPenalty ?? 0) + penalty
  drainPositiveBlockReward(context, penalty * 0.1)
  context.repetitionStreak = 0
  context.noveltyFlag = false
  console.warn(`[${label(context)}] Death detected via ${source} → -${penalty.toFixed(2)} reward penalty`)
}

function setupRewardTracking(context) {
  const { bot } = context

  bot.on('blockBreak', block => {
    if (!block || block.name === 'air') return
    const blockNameRaw = typeof block.name === 'string' ? block.name : ''
    const blockName = blockNameRaw.toLowerCase()
    const ore = isOreBlock(block)
    let value =
      blockName.includes('stone') ? 1.0 :
      blockName.includes('dirt') ? 0.5 :
      0.3
    if (ore) {
      value = Math.max(value, ORE_BLOCK_REWARD_BONUS)
    }
    addBlockReward(context, value, REWARD_SIGN.POSITIVE, ore ? 'ore-break' : 'block-break')
    if (context.mode === 'feral') {
      context.feralFury = Math.min(5, (context.feralFury ?? 0) + value * 0.2)
    }
    const blockPos = block.position
    if (blockPos) {
      const last = context.lastDigPosition
      const sameColumn =
        last &&
        Number.isInteger(last.x) &&
        Number.isInteger(last.y) &&
        Number.isInteger(last.z) &&
        last.x === blockPos.x &&
        last.z === blockPos.z
      if (sameColumn && last.y - blockPos.y === 1) {
        context.straightDownDigStreak = (context.straightDownDigStreak ?? 0) + 1
        if (context.straightDownDigStreak >= STRAIGHT_DOWN_DIG_THRESHOLD) {
          const excess = context.straightDownDigStreak - STRAIGHT_DOWN_DIG_THRESHOLD
          const penalty = STRAIGHT_DOWN_DIG_PENALTY + excess * STRAIGHT_DOWN_DIG_PENALTY_GROWTH
          addBlockReward(context, -penalty, REWARD_SIGN.NEGATIVE, 'dig-straight-down')
          console.log(
            `[${label(context)}] Straight-down digging penalty (${context.straightDownDigStreak}) → -${penalty.toFixed(2)} reward`
          )
        }
      } else {
        context.straightDownDigStreak = 0
      }
      context.lastDigPosition = { x: blockPos.x, y: blockPos.y, z: blockPos.z }
    }
    console.log(`[${label(context)}] Broke ${block.name} → +${value.toFixed(2)} reward`)
  })

  bot.on('diggingAborted', () => {
    addBlockReward(context, 0.1, REWARD_SIGN.NEGATIVE, 'dig-aborted')
    console.log(`[${label(context)}] Dig aborted → -0.1 penalty`)
    if (context.mode === 'feral') {
      context.feralFury = Math.max(0, (context.feralFury ?? 0) - 0.15)
    }
    context.straightDownDigStreak = 0
  })

  bot.on('playerCollect', (collector, collected) => {
    if (collector === bot.entity) {
      const count = collected?.metadata?.itemCount ?? collected?.count ?? 1
      const bonus = Math.max(0.3, (count || 1) * 0.15)
      addBlockReward(context, bonus, REWARD_SIGN.POSITIVE, 'item-collect')
      if (context.mode === 'feral') {
        context.feralFury = Math.max(0, (context.feralFury ?? 0) - 0.1)
      }
      const itemName = collected?.name ?? collected?.metadata?.item?.name
      const category = typeof itemName === 'string' ? categorizeResource(itemName) : null
      if (category && typeof context.resources[category] === 'number') {
        context.resources[category] += count
        GLOBAL_RESOURCE_POOL[category] += count
        noteResourceDiversity(category)
        registerContribution(context, count)
        updateCooperationScore(context)
      }
      if (typeof itemName === 'string' && isOreName(itemName)) {
        const oreBonus = Math.max(ORE_PICKUP_REWARD_BASE, count * ORE_PICKUP_REWARD_PER_ITEM)
        addBlockReward(context, oreBonus, REWARD_SIGN.POSITIVE, 'ore-collect')
        console.log(
          `[${label(context)}] Collected ore ${itemName} → +${oreBonus.toFixed(2)} reward`
        )
      }
      if (typeof itemName === 'string' && isAnimalFoodItem(itemName)) {
        const hungerBonusBase = isCriticalHunger(context) ? HUNGER_COLLECTION_REWARD * 1.5 : HUNGER_COLLECTION_REWARD
        if (isHungry(context)) {
          addBlockReward(context, hungerBonusBase, REWARD_SIGN.POSITIVE, 'item-collect-hungry')
          console.log(
            `[${label(context)}] Collected ${itemName} while hungry → +${hungerBonusBase.toFixed(2)} reward`
          )
        }
      }
      console.log(`[${label(context)}] Collected item → +${bonus.toFixed(2)} reward`)
    }
  })

  const grantAchievementReward = (payload, source = 'achievement') => {
    if (!payload && payload !== 0) return
    if (!(context.achievements instanceof Set)) {
      context.achievements = new Set()
    }
    let identifier = null
    if (typeof payload === 'string') {
      identifier = payload
    } else if (payload) {
      identifier =
        payload.id ??
        payload.advancement?.id ??
        payload.achievement ??
        payload.key ??
        payload.name ??
        (typeof payload.display?.title === 'string' ? payload.display.title : null) ??
        (typeof payload.title === 'string' ? payload.title : null) ??
        (typeof payload.advancement?.display?.title === 'string'
          ? payload.advancement.display.title
          : null)
    }
    if (!identifier || typeof identifier !== 'string') {
      return
    }
    const normalized = identifier.toLowerCase()
    if (context.achievements.has(normalized)) {
      return
    }
    context.achievements.add(normalized)
    context.achievementReward = (context.achievementReward ?? 0) + ACHIEVEMENT_REWARD_BONUS
    const title =
      (typeof payload?.display?.title === 'string' && payload.display.title) ||
      (typeof payload?.title === 'string' && payload.title) ||
      identifier
    console.log(
      `[${label(context)}] Unlocked ${title} (${source}) → +${ACHIEVEMENT_REWARD_BONUS.toFixed(2)} reward`
    )
  }

  bot.on('achievement', data => {
    try {
      grantAchievementReward(data, 'achievement')
    } catch (err) {
      console.warn(`[${label(context)}] Failed processing achievement reward:`, err)
    }
  })

  bot.on('advancementDone', data => {
    try {
      grantAchievementReward(data, 'advancement')
    } catch (err) {
      console.warn(`[${label(context)}] Failed processing advancement reward:`, err)
    }
  })

  bot.on('advancement', data => {
    try {
      grantAchievementReward(data, 'advancement')
    } catch (err) {
      console.warn(`[${label(context)}] Failed processing advancement event:`, err)
    }
  })

  bot.on('death', () => {
    applyDeathPenalty(context, 'death-event')
  })
}

function scheduleReconnect(context, reason = 'disconnect', delay = 5000) {
  if (!context || context.shuttingDown) return
  if (context.reconnectTimer) return
  context.running = false
  if (context.tickTimer) {
    clearTimeout(context.tickTimer)
    context.tickTimer = null
  }
  context.reconnecting = true
  context.reconnectAttempts = (context.reconnectAttempts ?? 0) + 1
  const backoff = Math.min(30000, Math.floor(delay * context.reconnectAttempts))
  console.warn(`[${label(context)}] Scheduling reconnect in ${backoff}ms (${reason}).`)
  context.reconnectTimer = setTimeout(() => {
    context.reconnectTimer = null
    try {
      context.bot?.removeAllListeners?.()
    } catch (err) {
      console.warn(`[${label(context)}] Failed to prune listeners before reconnect:`, err?.message ?? err)
    }
    try {
      const newBot = mineflayer.createBot({
        host: NETWORK_HOST,
        port: NETWORK_PORT,
        username: context.username
      })
      context.bot = newBot
      context.movementController?.reset()
      context.running = true
      context.reconnecting = false
      context.reconnectAttempts = 0
      setupBot(context)
    } catch (err) {
      console.error(`[${label(context)}] Reconnect attempt failed:`, err)
      scheduleReconnect(context, 'retry', backoff * 1.5)
    }
  }, backoff)
}

function setupBot(context) {
  const username = context.username
  console.log(`[${username}] Connecting to ${NETWORK_HOST}:${NETWORK_PORT}`)

  const { bot } = context

  bot.once('spawn', () => {
    console.log(`[${label(context)}] Spawned! Waiting for entity to initialize...`)

    context.reconnecting = false
    context.movementController?.reset()
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

  const handleDisconnect = reason => {
    console.warn(`[${label(context)}] Disconnected: ${reason}`)
    const normalized = typeof reason === 'string' ? reason.toLowerCase() : ''
    if (normalized.includes('death') || normalized.includes('died') || normalized.includes('killed')) {
      applyDeathPenalty(context, `disconnect:${reason}`)
    }
    scheduleReconnect(context, reason)
  }

  bot.once('end', () => handleDisconnect('end'))
  bot.on('kicked', r => handleDisconnect(`kicked: ${r}`))
  bot.on('error', e => {
    console.error(`[${label(context)}] Error:`, e)
    if (!context.reconnecting) {
      scheduleReconnect(context, e?.message ?? 'error')
    }
  })

  setupRewardTracking(context)
}

async function retireContext(context, reason = 'retire') {
  if (!context) return
  context.running = false
  context.shuttingDown = true
  if (context.tickTimer) {
    clearTimeout(context.tickTimer)
    context.tickTimer = null
  }
  if (context.reconnectTimer) {
    clearTimeout(context.reconnectTimer)
    context.reconnectTimer = null
  }
  try {
    context.bot?.removeAllListeners?.()
    context.bot?.quit?.(safeDisconnectReason(`Retire: ${reason}`))
  } catch (err) {
    console.warn(`[${label(context)}] Failed to retire bot:`, err?.message ?? err)
  }
  context.movementController?.reset()
  context.bot = null
  const idx = contexts.indexOf(context)
  if (idx >= 0) {
    contexts.splice(idx, 1)
  }
}

async function restartAllBots(reason = 'recovery', { resume } = {}) {
  const desiredRunning = resume ?? globalRunning
  const enforceLimit = typeof reason === 'string' && reason.startsWith('fatal')
  if (enforceLimit && fatalRecoveryAttempts >= FATAL_RECOVERY_MAX_ATTEMPTS) {
    console.error(
      `[Brain] Maximum recovery attempts (${FATAL_RECOVERY_MAX_ATTEMPTS}) reached. Skipping ${reason} restart.`
    )
    return
  }

  console.warn(`[Brain] Restarting bot population due to ${reason}.`)

  globalRunning = false

  const active = [...contexts]
  for (const ctx of active) {
    try {
      await retireContext(ctx, reason)
    } catch (err) {
      console.error(`[Brain] Failed retiring ${label(ctx)} during restart:`, err)
    }
  }

  contexts.length = 0

  try {
    await ensureBaselineReady()
  } catch (err) {
    if (isRemoteBrainUnavailableError(err)) {
      const status = getRemoteBrainStatus()
      console.warn(
        `[Brain] Baseline unavailable while restarting (${describeRemoteRetry(status)}). Proceeding with fresh contexts.`
      )
    } else {
      console.error('[Brain] Failed to prepare baseline during restart:', err)
    }
  }

  if (FATAL_RECOVERY_RESTART_DELAY_MS > 0) {
    await sleep(FATAL_RECOVERY_RESTART_DELAY_MS)
  }

  globalRunning = desiredRunning

  for (let i = 0; i < BOT_COUNT; i++) {
    const seedMutation = deriveMutationStddev(null, { jitter: true }) ?? NEW_BRAIN_MUTATION_STDDEV
    try {
      const ctx = createContext(i, { mutationStddev: seedMutation })
      if (ctx && !desiredRunning) {
        ctx.running = false
      }
    } catch (err) {
      console.error(`[Brain] Failed spawning context ${i} during restart:`, err)
    }
  }

  console.log(
    `[Brain] Restart complete (${contexts.length} bots active, running=${desiredRunning}).`
  )
}

async function handleFatalProcessError(source, error) {
  const now = Date.now()

  if (fatalRecoveryInProgress) {
    console.error(`[Brain] Additional fatal ${source} while recovery is in progress:`, error)
    return
  }

  if (now - lastFatalRecoveryAt > FATAL_RECOVERY_COOLDOWN_MS * 4) {
    fatalRecoveryAttempts = 0
  }

  if (fatalRecoveryAttempts >= FATAL_RECOVERY_MAX_ATTEMPTS) {
    console.error(
      `[Brain] Fatal ${source} detected but maximum recoveries reached (${FATAL_RECOVERY_MAX_ATTEMPTS}).`,
      error
    )
    return
  }

  if (now - lastFatalRecoveryAt < FATAL_RECOVERY_COOLDOWN_MS) {
    console.error(
      `[Brain] Fatal ${source} occurred within recovery cooldown (${FATAL_RECOVERY_COOLDOWN_MS}ms). Skipping restart.`,
      error
    )
    return
  }

  fatalRecoveryInProgress = true
  fatalRecoveryAttempts += 1
  lastFatalRecoveryAt = now

  const wasRunning = globalRunning
  console.error(
    `[Brain] Fatal ${source} detected. Attempting automated recovery (#${fatalRecoveryAttempts}).`,
    error
  )

  try {
    await safePersistState(`fatal-${source}`)
  } catch (persistErr) {
    console.error('[Brain] Recovery save step failed:', persistErr)
  }

  try {
    await restartAllBots(`fatal-${source}`, { resume: wasRunning })
  } catch (restartErr) {
    console.error('[Brain] Automated restart failed:', restartErr)
  } finally {
    fatalRecoveryInProgress = false
  }
}

function createContext(index, options = {}) {
  const parent = options.parent ?? null
  const partner = options.partner ?? null
  let mode = options.mode ?? null
  if (!mode) {
    if (parent?.mode === 'feral') {
      mode = 'feral'
    } else if ((contexts.length + 1) % FERAL_INTERVAL === 0) {
      mode = 'feral'
    } else {
      mode = 'civilized'
    }
  }

  let lineageBase = options.lineageBase ?? null
  if (!lineageBase) {
    if (parent?.lineage && partner?.lineage && partner.lineage !== parent.lineage) {
      lineageBase = combineLineageBases(parent.lineage, partner.lineage)
    } else if (parent?.lineage) {
      lineageBase = parent.lineage
    } else {
      lineageBase = generateRandomLineageBase()
    }
  }
  if (!lineageBase) {
    lineageBase = LINEAGE_ROOT_NAME
  }

  const identity = allocateLineageIdentity({ parentLineage: lineageBase, mode })
  const username = identity.username
  const bot = mineflayer.createBot({
    host: NETWORK_HOST,
    port: NETWORK_PORT,
    username
  })

  const emotion = new Float32Array(EMOTION_VECTOR_SIZE)
  for (let i = 0; i < EMOTION_VECTOR_SIZE; i++) {
    emotion[i] = MORALE_BASELINE
  }

  const explicitMutation = Number.isFinite(options.mutationStddev) && options.mutationStddev >= 0
    ? options.mutationStddev
    : null
  const primaryMutation = deriveMutationStddev(explicitMutation, { jitter: explicitMutation == null })
  const extraMutations = Array.isArray(options.extraMutations)
    ? options.extraMutations
        .map(value => deriveMutationStddev(value, { jitter: false }))
        .filter(value => Number.isFinite(value) && value > 0)
    : []

  const context = {
    id: index,
    username,
    lineage: identity.lineage,
    lineageOrdinal: identity.ordinal,
    lineagePrestige: getLineagePrestige(identity.lineage),
    birthOrder: birthCounter++,
    birthGeneration: baselineState.generation ?? 0,
    birthTick: baselineState.tickCount ?? 0,
    birthTime: Date.now(),
    mode,
    feralFury: 0,
    bot,
    brain: null,
    weightsSuspect: false,
    pendingWeightRecovery: null,
    weightRecoveryInFlight: null,
    lastWeightIssue: null,
    lastWeightRecovery: Date.now(),
    lastWeightRecoveryReason: 'init',
    weightSkipNotified: false,
    echoFallbackActive: false,
    pendingMutations: [],
    epsilon: EPSILON_START,
    epsilonBoost: 0,
    running: true,
    tickTimer: null,
    tickInFlight: false,
    movementController: null,
    registry: null,
    lastObs: null,
    lastAction: null,
    prevAction: null,
    lastPos: null,
    lastHealth: 20,
    lastFood: 20,
    lastInvTotal: 0,
    blockReward: 0,
    achievementReward: 0,
    lastDigPosition: null,
    straightDownDigStreak: 0,
    tickCount: 0,
    trainingSteps: 0,
    cumulativeReward: 0,
    generationTicks: 0,
    generationReward: 0,
    readyForSync: false,
    repetitionStreak: 0,
    rewardHistory: [],
    actionHistory: [],
    memoryRewards: [],
    memoryActions: [],
    actionValueMemory: new Map(),
    recentRewardAvg: 0,
    longRewardAvg: 0,
    rewardDrift: 0,
    rewardVolatility: 0,
    rewardSignStats: { corrections: 0, history: [] },
    cosinePenaltyScaling: { reasons: Object.create(null) },
    actionMemoryBest: 0,
    actionMemoryWorst: 0,
    actionCounts: new Map(),
    behaviorEntropy: 0,
    visitedStates: new Set(),
    visitedBlocks: new Set(),
    visitedBiomes: new Set(),
    noveltyCount: 0,
    noveltyFlag: false,
    deathPenalty: 0,
    damageDebt: 0,
    recentDamage: 0,
    lastDeathAt: 0,
    resources: {
      wood: 0,
      stone: 0,
      ore: 0,
      crafted: 0
    },
    resourceLedger: {
      contributed: 0,
      withdrawn: 0
    },
    cooperationScore: 0,
    skillChains: new Map(),
    currentChainScore: 0,
    stagnation: {
      active: false,
      streak: 0,
      lastMutation: baselineState.generation ?? 0
    },
    rewardTrend: {
      generationHistory: [],
      downwardStreak: 0,
      flaggedDownward: false,
      lastSlope: 0,
      lastWindow: [],
      lastNetDrop: 0,
      lastUpdatedGeneration: baselineState.generation ?? 0
    },
    morale: {
      value: MORALE_BASELINE,
      frustration: 0,
      sharpness: 0.5,
      successStreak: 0,
      failureStreak: 0
    },
    emotion,
    reconnectAttempts: 0,
    reconnectTimer: null,
    reconnecting: false,
    shuttingDown: false,
    waitingForBrain: null,
    remoteBrain: {
      offlineNotified: false,
      lastMessage: null,
      nextLogAt: 0
    },
    achievements: new Set(),
    parents: {
      primary: parent?.username ?? null,
      primaryLineage: parent?.lineage ?? null,
      partner: partner?.username ?? null,
      partnerLineage: partner?.lineage ?? null
    },
    nonFiniteTracker: null,
    communication: createCommunicationState()
  }

  resetNonFiniteTracker(context)

  context.movementController = createMovementController(context)

  if (Number.isFinite(primaryMutation) && primaryMutation > 0) {
    enqueuePendingMutation(context, primaryMutation)
  }
  for (const extra of extraMutations) {
    enqueuePendingMutation(context, extra)
  }

  setupBot(context)
  contexts.push(context)
  const parentLineageLabel = parent?.lineage
    ? `${parent.lineage}-${romanNumeral(parent.lineageOrdinal ?? 1)}`
    : null
  const partnerLineageLabel = partner?.lineage
    ? `${partner.lineage}-${romanNumeral(partner.lineageOrdinal ?? 1)}`
    : null
  const crossoverLabel =
    partnerLineageLabel && parentLineageLabel && partnerLineageLabel !== parentLineageLabel
      ? ` via crossover with ${partner?.username ?? partnerLineageLabel}`
      : ''
  const lineageLabel = `${context.lineage}-${romanNumeral(context.lineageOrdinal)}`
  const parentLabel = parent?.username ?? 'baseline'
  console.log(
    `[${label(context)}] Born from lineage ${lineageLabel} (${context.mode}). Parent ${parentLabel}${
      crossoverLabel || ''
    }.`
  )
  return context
}

for (let i = 0; i < BOT_COUNT; i++) {
  const seedMutation = deriveMutationStddev(null, { jitter: true }) ?? NEW_BRAIN_MUTATION_STDDEV
  createContext(i, { mutationStddev: seedMutation })
}

process.stdin.resume()
process.stdin.setEncoding('utf8')
console.log('[Brain] Type "pause", "resume", "save", "restart", or "exit".')

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
  } else if (cmd === 'restart') {
    const wasRunning = globalRunning
    console.log('[Brain] Manual restart requested...')
    try {
      await safePersistState('manual-restart')
    } catch (err) {
      console.error('[Brain] Manual restart save failed:', err)
    }
    try {
      await restartAllBots('manual-restart', { resume: wasRunning })
    } catch (err) {
      console.error('[Brain] Manual restart failed:', err)
    }
  } else if (cmd === 'save') {
    console.log('[Brain] Manual save requested...')
    scheduleBaselineSave('manual')
  } else if (['exit', 'quit', 'stop'].includes(cmd)) {
    console.log('[Brain] Saving model + shutting down...')
    globalRunning = false
    for (const ctx of contexts) {
      ctx.running = false
      ctx.shuttingDown = true
      if (ctx.tickTimer) {
        clearTimeout(ctx.tickTimer)
        ctx.tickTimer = null
      }
    }
    await flushPendingSave()
    if (isRemoteBrainConnected()) {
      try {
        await ensureBaselineReady()
        await persistBaseline('shutdown')
      } catch (err) {
        if (isRemoteBrainUnavailableError(err)) {
          const status = getRemoteBrainStatus()
          console.warn(`[Brain] Skipping shutdown save: remote brain unavailable (${describeRemoteRetry(status)}).`)
        } else {
          console.error('[Brain] Failed to persist baseline during shutdown:', err)
        }
      }
    } else {
      const status = getRemoteBrainStatus()
      console.warn(`[Brain] Remote brain unavailable during shutdown (${describeRemoteRetry(status)}). Skipping save.`)
    }
    for (const ctx of contexts) {
      try {
        ctx.bot?.quit?.(safeDisconnectReason('Manual shutdown'))
      } catch (err) {
        console.warn(`[${label(ctx)}] Failed to quit bot during shutdown:`, err)
      }
    }
    stopMovementUpdateLoop()
    await shutdownBrainWorkerPool()
    process.exit(0)
  }
})

process.on('uncaughtException', err => {
  handleFatalProcessError('uncaughtException', err).catch(recoveryErr => {
    console.error('[Brain] Uncaught exception recovery handler failed:', recoveryErr)
  })
})

process.on('unhandledRejection', reason => {
  const error = reason instanceof Error ? reason : new Error(`Unhandled rejection: ${String(reason)}`)
  handleFatalProcessError('unhandledRejection', error).catch(recoveryErr => {
    console.error('[Brain] Unhandled rejection recovery handler failed:', recoveryErr)
  })
})

async function gracefulShutdown(reason = 'signal') {
  try {
    console.log(`[Brain] Caught ${reason}. Saving before exit...`)
    globalRunning = false
    for (const ctx of contexts) {
      ctx.running = false
      ctx.shuttingDown = true
      if (ctx.tickTimer) {
        clearTimeout(ctx.tickTimer)
        ctx.tickTimer = null
      }
    }
    await flushPendingSave()
    if (isRemoteBrainConnected()) {
      try {
        await ensureBaselineReady()
        await persistBaseline(reason)
      } catch (err) {
        if (isRemoteBrainUnavailableError(err)) {
          const status = getRemoteBrainStatus()
          console.warn(`[Brain] Skipping ${reason} save: remote brain unavailable (${describeRemoteRetry(status)}).`)
        } else {
          throw err
        }
      }
    } else {
      const status = getRemoteBrainStatus()
      console.warn(`[Brain] Remote brain unavailable during ${reason} shutdown (${describeRemoteRetry(status)}). Skipping save.`)
    }
    for (const ctx of contexts) {
      try {
        ctx.bot?.quit?.(safeDisconnectReason(`Shutdown: ${reason}`))
      } catch (err) {
        console.warn(`[${label(ctx)}] Failed to quit bot during shutdown:`, err)
      }
    }
  } catch (err) {
    console.error('[Brain] Failed during graceful shutdown:', err)
  } finally {
    stopMovementUpdateLoop()
    try {
      await shutdownBrainWorkerPool()
    } catch (poolErr) {
      console.warn('[Brain] Failed to shutdown brain worker pool:', poolErr)
    }
    process.exit(0)
  }
}

process.on('SIGINT', () => gracefulShutdown('SIGINT'))
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'))
