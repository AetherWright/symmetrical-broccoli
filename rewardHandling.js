export const REWARD_SIGN = Object.freeze({
  POSITIVE: 'positive',
  NEGATIVE: 'negative',
  EITHER: 'either'
})

export function createRewardHandling({
  maxRewardMagnitude,
  baseObsFeatures,
  rewardEventBus,
  rewardEventDecayMs,
  rewardEventValueClamp,
  rewardEventThreshold,
  rewardEventRefreshMs,
  rewardEventBusLimit,
  rewardEventRange,
  deathRewardPenalty,
  applyCosinePenaltyScaling,
  limitPositive,
  label,
  noteGroupInteraction,
  ensureTemporalMemoryState,
  decayGroupAffinity,
  updateCooperationScore,
  computeGroupAffinityBonus,
  computeCrowdingPenalty,
  updateStagnation,
  updateSkillChains,
  adjustMorale,
  getLineagePrestige,
  rewardProfile,
  damageDebtDecay,
  damageRecentDecay,
  damageMemoryClamp,
  damageDebtWeight,
  damageDebtPenalty,
  getRewardKalmanFilterState
}) {
  function clampReward(value) {
    if (!Number.isFinite(value)) {
      return 0
    }
    const limit = maxRewardMagnitude
    if (value > limit) return limit
    if (value < -limit) return -limit
    return value
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
      reward: limitPositive(accumulator.reward, maxRewardMagnitude),
      penalty: limitPositive(accumulator.penalty, maxRewardMagnitude)
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
      noteRewardHighlight(context, reason, adjusted)
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

  function pruneRewardEventBus(now = Date.now()) {
    const bus = rewardEventBus
    if (!bus) return
    if (now < bus.lastPrune + 200) {
      return
    }
    bus.lastPrune = now
    const cutoff = now - rewardEventDecayMs
    if (!Number.isFinite(cutoff) || cutoff <= 0) {
      return
    }
    if (!Array.isArray(bus.events) || bus.events.length === 0) {
      bus.events = []
      return
    }
    const filtered = []
    for (const event of bus.events) {
      const createdAt = Number(event?.createdAt)
      if (!Number.isFinite(createdAt) || createdAt < cutoff) {
        continue
      }
      filtered.push(event)
    }
    if (filtered.length > rewardEventBusLimit) {
      bus.events = filtered.slice(filtered.length - rewardEventBusLimit)
    } else {
      bus.events = filtered
    }
  }

  function refreshRewardHighlight(context, now = Date.now()) {
    if (!context?.rewardHighlight) {
      return
    }
    const createdAt = Number(context.rewardHighlight.createdAt)
    if (!Number.isFinite(createdAt) || now - createdAt > rewardEventDecayMs) {
      context.rewardHighlight = null
    }
  }

  function broadcastRewardHighlight(context, now = Date.now()) {
    if (!context?.rewardHighlight) return
    const highlight = context.rewardHighlight
    const amount = limitPositive(Number(highlight.amount) || 0, rewardEventValueClamp)
    if (amount <= 0) return
    const position = context.bot?.entity?.position
    if (!position) return
    const record = {
      senderId: context.id,
      lineage: context.lineage ?? null,
      mode: context.mode ?? null,
      reason: highlight.reason ?? 'unspecified',
      amount,
      createdAt: Number(highlight.createdAt) || now,
      position: {
        x: Number(position.x) || 0,
        y: Number(position.y) || 0,
        z: Number(position.z) || 0
      }
    }
    pruneRewardEventBus(now)
    rewardEventBus.events.push(record)
    if (rewardEventBus.events.length > rewardEventBusLimit) {
      rewardEventBus.events.splice(
        0,
        Math.max(0, rewardEventBus.events.length - rewardEventBusLimit)
      )
    }
  }

  function noteRewardHighlight(context, reason, amount, now = Date.now()) {
    if (!context || !Number.isFinite(amount) || amount <= 0) {
      return
    }
    if (amount < rewardEventThreshold) {
      return
    }
    const normalizedReason = typeof reason === 'string' && reason ? reason : 'unspecified'
    const clampedAmount = limitPositive(amount, rewardEventValueClamp)
    const highlight = context.rewardHighlight
    const shouldReplace =
      !highlight ||
      !Number.isFinite(highlight.amount) ||
      clampedAmount > highlight.amount * 1.05 ||
      now - (Number(highlight.createdAt) || 0) >= rewardEventRefreshMs
    if (!shouldReplace) {
      return
    }
    context.rewardHighlight = {
      amount: clampedAmount,
      reason: normalizedReason,
      createdAt: now
    }
    broadcastRewardHighlight(context, now)
  }

  function hashRewardReason(reason) {
    if (!reason) return 0
    const text = String(reason)
    let hash = 0
    for (let i = 0; i < text.length; i++) {
      hash = (hash * 33 + text.charCodeAt(i)) >>> 0
    }
    const bucket = hash % 997
    return bucket / 996
  }

  function ensureRewardSignalState(context) {
    if (!context) return null
    if (!context.rewardSignal) {
      context.rewardSignal = { amount: 0, reason: 0, freshness: 0, updatedAt: 0 }
    }
    return context.rewardSignal
  }

  function updateRewardEventAwareness(context) {
    const signal = ensureRewardSignalState(context)
    if (!signal || !context?.bot?.entity?.position) {
      if (signal) {
        signal.amount = 0
        signal.reason = 0
        signal.freshness = 0
        signal.updatedAt = Date.now()
      }
      return signal
    }
    const now = Date.now()
    pruneRewardEventBus(now)
    const position = context.bot.entity.position
    let topIntensity = 0
    let topReason = 0
    let topFreshness = 0
    for (const event of rewardEventBus.events) {
      if (!event) continue
      const dt = now - Number(event.createdAt)
      if (!Number.isFinite(dt) || dt < 0 || dt > rewardEventDecayMs) continue
      let intensity = limitPositive(Number(event.amount) || 0, rewardEventValueClamp)
      if (intensity <= 0) continue
      const timeDecay = Math.max(0, 1 - dt / rewardEventDecayMs)
      intensity *= timeDecay
      const eventPos = event.position
      if (eventPos && position) {
        const dx = Number(eventPos.x) - Number(position.x)
        const dy = Number(eventPos.y) - Number(position.y)
        const dz = Number(eventPos.z) - Number(position.z)
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz)
        if (!Number.isFinite(dist) || dist > rewardEventRange) {
          continue
        }
        const spatialDecay = Math.max(0, 1 - dist / rewardEventRange)
        intensity *= spatialDecay
      } else if (event.senderId !== context.id) {
        continue
      }
      if (event.senderId !== context.id && context.mode && event.mode) {
        const sameMode = event.mode === context.mode
        noteGroupInteraction(context, sameMode, intensity, 'reward-highlight')
      }
      if (intensity <= topIntensity) continue
      topIntensity = intensity
      topReason = hashRewardReason(event.reason)
      topFreshness = timeDecay
    }
    signal.amount = topIntensity
    signal.reason = Math.max(0, Math.min(1, topReason))
    signal.freshness = Math.max(0, Math.min(1, topFreshness))
    signal.updatedAt = now
    refreshRewardHighlight(context, now)
    return signal
  }

  function computeReward(context) {
    const frame =
      (context.currentObservationFrame instanceof Float32Array && context.currentObservationFrame.length >= baseObsFeatures)
        ? context.currentObservationFrame
        : context.lastObservationFrame
    if (!(frame instanceof Float32Array)) {
      return createRewardAccumulator()
    }
    let reward = createRewardAccumulator()

    ensureTemporalMemoryState(context)
    decayGroupAffinity(context)
    context.damageDebt = limitPositive((context.damageDebt ?? 0) * damageDebtDecay, damageMemoryClamp)
    context.recentDamage = limitPositive((context.recentDamage ?? 0) * damageRecentDecay, damageMemoryClamp)

    const pos = { x: frame[0], y: frame[1], z: frame[2] }
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

    const health = frame[8]
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
        context.damageDebt = limitPositive(context.damageDebt + damage * damageDebtWeight, damageMemoryClamp)
        context.recentDamage = limitPositive(context.recentDamage + damage, damageMemoryClamp)
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

    const food = frame[9]
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

    const invTotal = frame[18]
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

    const nearestDist = frame[16]
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

    const crowdPenalty = computeCrowdingPenalty(context)
    if (crowdPenalty > 0) {
      reward = applyRewardComponent(reward, -crowdPenalty, REWARD_SIGN.NEGATIVE, context, 'crowding')
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
        Math.min(2, context.damageDebt * damageDebtPenalty),
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

    const affinityBonus = computeGroupAffinityBonus(context)
    if (affinityBonus !== 0) {
      reward = applyRewardComponent(
        reward,
        affinityBonus,
        REWARD_SIGN.EITHER,
        context,
        'group-affinity'
      )
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

  function applyDeathPenalty(context, source = 'unknown') {
    if (!context) return
    const now = Date.now()
    if (context.lastDeathAt && now - context.lastDeathAt < 1000) {
      return
    }
    context.lastDeathAt = now
    const penalty = Math.max(5, deathRewardPenalty)
    context.deathPenalty = (context.deathPenalty ?? 0) + penalty
    drainPositiveBlockReward(context, penalty * 0.1)
    context.repetitionStreak = 0
    context.noveltyFlag = false
    console.warn(`[${label(context)}] Death detected via ${source} → -${penalty.toFixed(2)} reward penalty`)
  }

  function clearIncomingRewards(context, reason = 'death') {
    if (!context) return
    context.blockReward = 0
    context.achievementReward = 0
    context.rewardHighlight = null
    context.rewardSignal = null
    const kalmanFactory = typeof getRewardKalmanFilterState === 'function'
      ? getRewardKalmanFilterState()
      : null
    if (typeof kalmanFactory === 'function') {
      context.rewardKalman = kalmanFactory()
    }
    context.repetitionStreak = 0
    context.noveltyFlag = false
    console.warn(`[${label(context)}] Cleared pending rewards after ${reason}.`)
  }

  return {
    clampReward,
    recordRewardSignCorrection,
    ensureRewardSign,
    createRewardAccumulator,
    ensureRewardAccumulator,
    finalizeRewardAccumulator,
    applyRewardComponent,
    addBlockReward,
    drainPositiveBlockReward,
    pruneRewardEventBus,
    refreshRewardHighlight,
    broadcastRewardHighlight,
    noteRewardHighlight,
    hashRewardReason,
    ensureRewardSignalState,
    updateRewardEventAwareness,
    computeReward,
    applyDeathPenalty,
    clearIncomingRewards
  }
}
