export function createTickScheduler({ runTick, isGlobalRunning, isContextRunning }) {
  const pendingSystemTicks = new Map()
  let systemTickHandle = null
  let systemTickInFlight = false

  function scheduleSystemTick() {
    if (!isGlobalRunning()) return
    if (systemTickHandle || systemTickInFlight) {
      return
    }

    systemTickHandle = setTimeout(() => {
      systemTickHandle = null
      runSystemTickQueue().catch(err => {
        console.error('[Brain] System tick error:', err)
      })
    }, 0)
  }

  async function runSystemTickQueue() {
    if (systemTickInFlight) return
    systemTickInFlight = true

    try {
      while (isGlobalRunning()) {
        let nextContext = null
        for (const [candidate] of pendingSystemTicks) {
          if (!isContextRunning(candidate) || candidate.tickInFlight) {
            pendingSystemTicks.delete(candidate)
            continue
          }
          nextContext = candidate
          break
        }

        if (!nextContext) {
          break
        }

        pendingSystemTicks.delete(nextContext)

        try {
          await runTick(nextContext)
        } catch (err) {
          console.error(`[${nextContext?.username ?? 'Unknown'}] Tick scheduling error:`, err)
        }

        if (!isGlobalRunning()) {
          break
        }
      }
    } finally {
      systemTickInFlight = false
      if (pendingSystemTicks.size > 0 && isGlobalRunning()) {
        scheduleSystemTick()
      }
    }
  }

  function scheduleTick(context, reason = 'manual') {
    if (!context) return
    if (!isGlobalRunning() || !isContextRunning(context)) return

    if (context.tickInFlight) {
      if (reason !== 'action-complete' && !context.pendingEnvironmentTick) {
        context.pendingEnvironmentTick = true
        context.pendingEnvironmentReason = reason
      }
      return
    }

    const existingReason = pendingSystemTicks.get(context)
    if (existingReason) {
      if (existingReason !== 'action-complete' && reason === 'action-complete') {
        pendingSystemTicks.set(context, reason)
      }
      return
    }

    pendingSystemTicks.set(context, reason)
    scheduleSystemTick()
  }

  function cancelPendingTick(context) {
    if (!context) return
    const removed = pendingSystemTicks.delete(context)
    if (removed && pendingSystemTicks.size === 0 && systemTickHandle && !systemTickInFlight) {
      clearTimeout(systemTickHandle)
      systemTickHandle = null
    }
  }

  function isTickPending(context) {
    if (!context) return false
    return pendingSystemTicks.has(context)
  }

  function clearAllPendingTicks() {
    pendingSystemTicks.clear()
    if (systemTickHandle && !systemTickInFlight) {
      clearTimeout(systemTickHandle)
      systemTickHandle = null
    }
  }

  function noteEnvironmentChange(context, reason = 'environment') {
    if (!context) return
    if (context.waitingForBrain && reason !== 'remote-brain-online') {
      return
    }
    scheduleTick(context, reason)
  }

  function bindEnvironmentTriggers(context) {
    if (!context?.bot) return

    unbindEnvironmentTriggers(context)

    const { bot } = context
    const listeners = [
      ['move', () => noteEnvironmentChange(context, 'self-move')],
      ['blockUpdate', () => noteEnvironmentChange(context, 'block-update')],
      ['chunkColumnLoad', () => noteEnvironmentChange(context, 'chunk-load')],
      ['chunkColumnUnload', () => noteEnvironmentChange(context, 'chunk-unload')],
      ['entitySpawn', () => noteEnvironmentChange(context, 'entity-spawn')],
      ['entityGone', () => noteEnvironmentChange(context, 'entity-gone')],
      ['entityHurt', () => noteEnvironmentChange(context, 'entity-hurt')],
      ['death', () => noteEnvironmentChange(context, 'death')],
      ['health', () => noteEnvironmentChange(context, 'health-change')],
      ['experience', () => noteEnvironmentChange(context, 'experience-change')],
      ['collect', () => noteEnvironmentChange(context, 'collect')],
      ['forcedMove', () => noteEnvironmentChange(context, 'forced-move')]
    ]

    context.environmentListeners = listeners
    for (const [event, handler] of listeners) {
      bot.on(event, handler)
    }
  }

  function unbindEnvironmentTriggers(context) {
    if (!context?.bot) return

    const listeners = Array.isArray(context.environmentListeners)
      ? context.environmentListeners
      : []

    for (const [event, handler] of listeners) {
      context.bot.off(event, handler)
    }

    context.environmentListeners = []
  }

  return {
    scheduleTick,
    cancelPendingTick,
    isTickPending,
    clearAllPendingTicks,
    noteEnvironmentChange,
    bindEnvironmentTriggers,
    unbindEnvironmentTriggers
  }
}
