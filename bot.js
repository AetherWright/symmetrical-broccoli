import mineflayer from 'mineflayer'
import { Vec3 } from 'vec3'
import fs from 'node:fs'
import path from 'node:path'
import { pipeline } from 'node:stream'
import { promisify } from 'node:util'
import zlib from 'node:zlib'
import {
  createBrain,
  chooseAction,
  trainBrain,
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

// ----------------------------
// CONFIG
// ----------------------------
const MC_HOST = 'localhost'
const MC_PORT = 25565
const BOT_COUNT = Math.max(2, parseInt(process.env.BOT_COUNT ?? '10', 10))
const GENERATION_TICKS = Math.max(50, parseInt(process.env.GENERATION_TICKS ?? '200', 10))
const ROCK_PARTS = ['Rock', 'Stone', 'Grav', 'Ore', 'Pebble', 'Granite', 'Basalt', 'Iron', 'Coal', 'Quartz']
const SUFFIXES = ['son', 'grip', 'deep', 'delver', 'breaker', 'forge', 'drill', 'hammer', 'core', 'blast']
const PREFIXES = ['', 'Mc', 'Von', 'De', "O'", 'El']

const LINEAGE_ROOT_NAME = 'VonBasaltdeep'
const FERAL_DEFAULT_RATIO = Math.min(0.5, Math.max(0, Number.parseFloat(process.env.FERAL_RATIO ?? '0.2')))
const FERAL_INTERVAL = Math.max(2, Math.round(1 / (FERAL_DEFAULT_RATIO || 0.2)))
const EMOTION_DECAY = 0.92
const MORALE_BASELINE = 0.55
const REWARD_MUTATION_INTERVAL = Math.max(500, parseInt(process.env.REWARD_MUTATION_INTERVAL ?? '2500', 10))
const REWARD_MUTATION_JITTER = Math.max(100, parseInt(process.env.REWARD_MUTATION_JITTER ?? '600', 10))
const CROSSOVER_GENERATION_INTERVAL = Math.max(1, parseInt(process.env.CROSSOVER_INTERVAL ?? '5', 10))

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
  }
  tracker.offlineNotified = false
  tracker.lastMessage = null
  tracker.nextLogAt = 0
  context.waitingForBrain = false
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
  ...Object.keys(CRAFTING_ACTIONS)
]

const TICK_RATE = 1000
const EPSILON_START = 0.25
const EPSILON_MIN = 0.05
const EPSILON_DECAY = 0.999
const EPSILON_STAGNATION_BOOST = 0.4
const EPSILON_BOOST_DECAY = 0.995
const SAVE_INTERVAL_TICKS = 40
const SAVE_INTERVAL_MS = 60 * 1000
const CHECKPOINT_DIR = DEFAULT_BRAIN_DIR
const EMOTION_VECTOR_SIZE = 3
const OBS_SIZE = 44
const NOVELTY_HASH_PRECISION = 2
const NOVELTY_TARGET = 500
const STAGNATION_WINDOW = 40
const STAGNATION_VARIANCE_THRESHOLD = 0.0025
const STAGNATION_MUTATION_THRESHOLD = 3
const DIVERSITY_WINDOW = 60
const MAX_BOTS = Math.max(BOT_COUNT, parseInt(process.env.BOT_MAX ?? '8', 10))
const MIN_BOTS = Math.max(2, parseInt(process.env.BOT_MIN ?? '2', 10))
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
      lastSeenGeneration: baselineState.generation
    }
    lineageStats.set(lineageName, stats)
  }
  return stats
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

function categorizeResource(name) {
  if (!name) return null
  if (name.includes('log') || name.includes('wood')) return 'wood'
  if (name.includes('stone') || name.includes('cobblestone') || name.includes('gravel')) return 'stone'
  if (name.includes('ore') || name.includes('ingot') || name.includes('coal') || name.includes('iron')) return 'ore'
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
      ctx.blockReward += bonus
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
  const bounded = Math.max(-2, Math.min(2, reward))

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
  let modified = reward * moraleBoost * frustrationPenalty * sharpnessBoost * rewardProfile.morale

  if (!Number.isFinite(modified)) {
    modified = reward
  }

  const emotion = ensureEmotionVector(context)
  const targets = [morale.value, morale.frustration, morale.sharpness ?? 0.5]
  for (let i = 0; i < EMOTION_VECTOR_SIZE; i++) {
    emotion[i] = emotion[i] * EMOTION_DECAY + targets[i] * (1 - EMOTION_DECAY)
  }

  return modified
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

  return obs
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
    context.blockReward += config.reward ?? 0.4
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
    context.blockReward -= 0.03
  }
}

async function performMining(context, { forward = false, strafe = 0 } = {}) {
  const { bot } = context
  const target = bot.blockAtCursor(5)
  if (!target) {
    context.blockReward -= 0.02
    return
  }

  if (forward) {
    bot.setControlState('forward', true)
  }
  if (strafe < 0) {
    bot.setControlState('left', true)
  } else if (strafe > 0) {
    bot.setControlState('right', true)
  }

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
    context.blockReward += 0.5
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
    context.blockReward -= 0.05
    if (context.mode === 'feral') {
      context.feralFury = Math.max(0, (context.feralFury ?? 0) - 0.2)
    }
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
        const entity = bot.nearestEntity()
        if (entity) {
          try {
            await bot.attack(entity)
            context.blockReward += 0.3
            if (context.mode === 'feral') {
              context.feralFury = Math.min(5, (context.feralFury ?? 0) + 0.6)
            }
          } catch (err) {
            console.warn(`[${label(context)}] Attack failed:`, err?.message ?? err)
            context.blockReward -= 0.05
            if (context.mode === 'feral') {
              context.feralFury = Math.max(0, (context.feralFury ?? 0) - 0.1)
            }
          }
        } else {
          context.blockReward -= 0.01
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
              context.blockReward += 0.25
              registerWithdrawal(context, 1)
              updateCooperationScore(context)
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
                registerWithdrawal(context, 1)
                updateCooperationScore(context)
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
      case 'build_forward': {
        const target = getTargetBlock()
        if (target) {
          const success = await equipPlaceableBlock(context)
          if (success) {
            try {
              await bot.placeBlock(target, new Vec3(1, 0, 0))
              bot.setControlState('forward', true)
              await sleep(200)
              context.blockReward += 0.22
              registerWithdrawal(context, 1)
              updateCooperationScore(context)
            } catch (err) {
              console.warn(`[${label(context)}] Build forward failed:`, err?.message ?? err)
              context.blockReward -= 0.02
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

  if (context.lastAction != null) {
    if (context.prevAction === context.lastAction) {
      context.repetitionStreak += 1
    } else {
      context.repetitionStreak = 0
    }
    const fatiguePenalty = Math.min(0.6, context.repetitionStreak * 0.05)
    reward -= fatiguePenalty
  }

  if (context.noveltyFlag) {
    reward += 0.12 * rewardProfile.novelty
  }

  updateCooperationScore(context)
  reward += Math.max(-0.3, Math.min(0.3, context.cooperationScore * 0.5)) * rewardProfile.cooperation
  reward += (context.behaviorEntropy - 0.5) * 0.1 * rewardProfile.entropy
  reward += Math.min(0.25, context.currentChainScore * 0.2) * rewardProfile.skill

  const consumedBlockReward = context.blockReward
  reward += consumedBlockReward * rewardProfile.resource
  context.blockReward = 0

  const lineageBonus = getLineagePrestige(context.lineage) * 0.1 * rewardProfile.lineage
  reward += lineageBonus

  if (context.mode === 'feral') {
    const fury = context.feralFury ?? 0
    reward += Math.min(0.5, fury * 0.08) * rewardProfile.feral
    reward -= Math.max(0, context.cooperationScore) * 0.2
    context.feralFury = Math.max(0, fury * 0.92)
  } else {
    context.feralFury = Math.max(0, (context.feralFury ?? 0) * 0.85)
  }

  reward -= 0.02

  reward = adjustMorale(context, reward)

  updateStagnation(context, reward)
  updateSkillChains(context, reward)

  context.lineagePrestige = getLineagePrestige(context.lineage)

  context.lastPos = { ...pos }
  return reward
}

async function tickLoop(context) {
  if (!globalRunning || !context.running || context.tickInFlight) return
  context.tickInFlight = true

  let remoteUnavailable = false
  let remoteIssue = null
  try {
    const status = getRemoteBrainStatus()
    if (!status.connected) {
      remoteUnavailable = true
      remoteIssue = status
      return
    }

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
      trained = await trainBrain(
        context.brain,
        context.lastObs,
        context.lastAction,
        reward,
        observation
      )
    }

    const { action, prediction } = await chooseAction(context.brain, observation, effectiveEpsilon)
    await executeAction(context, action)

    const actionLabel = ACTIONS[action] ?? String(action)
    updateBehaviorEntropy(context, actionLabel)

    context.prevAction = context.lastAction
    context.lastObs = observation
    context.lastAction = action
    context.lastPrediction = Array.isArray(prediction) ? prediction : null

    context.tickCount += 1
    context.generationTicks += 1
    context.cumulativeReward += reward
    context.generationReward += reward
    baselineState.tickCount += 1
    baselineState.cumulativeReward += reward
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

    const predictionNote = context.lastPredictionError != null
      ? ` | PredErr: ${context.lastPredictionError.toFixed(3)}`
      : ''
    console.log(
      `[${label(context)}] Tick done | Reward: ${reward.toFixed(3)} | Eps: ${effectiveEpsilon.toFixed(3)} | Entropy: ${context.behaviorEntropy.toFixed(2)}${predictionNote}`
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

    const topReward = sorted[0]?.generationReward ?? -Infinity
    const averageReward = contexts.reduce((sum, ctx) => sum + ctx.generationReward, 0) / contexts.length

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

    const retireList = contractionCount > 0 ? sorted.slice(-contractionCount) : []

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
        .filter(ctx => !retireReasons.has(ctx) && shouldCullForDowntrend(ctx))
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

    for (let i = 0; i < expansionCount; i++) {
      const parentCandidate = topTwo[i % (topTwo.length || 1)] ?? sorted[0] ?? null
      createContext(contexts.length + i, { parent: parentCandidate })
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
    if (context.mode === 'feral') {
      context.feralFury = Math.min(5, (context.feralFury ?? 0) + value * 0.2)
    }
    console.log(`[${label(context)}] Broke ${block.name} → +${value.toFixed(2)} reward`)
  })

  bot.on('diggingAborted', () => {
    context.blockReward -= 0.1
    console.log(`[${label(context)}] Dig aborted → -0.1 penalty`)
    if (context.mode === 'feral') {
      context.feralFury = Math.max(0, (context.feralFury ?? 0) - 0.15)
    }
  })

  bot.on('playerCollect', (collector, collected) => {
    if (collector === bot.entity) {
      const count = collected?.metadata?.itemCount ?? collected?.count ?? 1
      const bonus = Math.max(0.3, (count || 1) * 0.15)
      context.blockReward += bonus
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
      console.log(`[${label(context)}] Collected item → +${bonus.toFixed(2)} reward`)
    }
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
  context.bot = null
  const idx = contexts.indexOf(context)
  if (idx >= 0) {
    contexts.splice(idx, 1)
  }
}

function createContext(index, options = {}) {
  const parent = options.parent ?? null
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

  const identity = allocateLineageIdentity({ parentLineage: parent?.lineage ?? LINEAGE_ROOT_NAME, mode })
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
    epsilon: EPSILON_START,
    epsilonBoost: 0,
    running: true,
    tickTimer: null,
    tickInFlight: false,
    registry: null,
    lastObs: null,
    lastAction: null,
    prevAction: null,
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
    readyForSync: false,
    repetitionStreak: 0,
    rewardHistory: [],
    actionHistory: [],
    actionCounts: new Map(),
    behaviorEntropy: 0,
    visitedStates: new Set(),
    visitedBlocks: new Set(),
    visitedBiomes: new Set(),
    noveltyCount: 0,
    noveltyFlag: false,
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
    waitingForBrain: false,
    remoteBrain: {
      offlineNotified: false,
      lastMessage: null,
      nextLogAt: 0
    }
  }

  setupBot(context)
  contexts.push(context)
  console.log(`[${label(context)}] Born from lineage ${context.lineage}-${romanNumeral(context.lineageOrdinal)} (${context.mode}).`)
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
    process.exit(0)
  }
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
    process.exit(0)
  }
}

process.on('SIGINT', () => gracefulShutdown('SIGINT'))
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'))
