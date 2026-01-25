export function createRewardKalmanHelpers({
  processNoise,
  measurementNoise,
  measurementGrowth,
  initialVariance,
  minVariance,
  maxVariance,
  minDeterminant,
  maxRewardMagnitude,
  clampReward,
  limitPositive,
  monotonicNow
}) {
  function clampKalmanVariance(value) {
    if (!Number.isFinite(value)) {
      return maxVariance
    }
    if (value < minVariance) {
      return minVariance
    }
    if (value > maxVariance) {
      return maxVariance
    }
    return value
  }

  function createRewardKalmanFilterState() {
    const variance = Math.min(maxVariance, Math.max(minVariance, initialVariance))
    return {
      state: new Float64Array([0, 0]),
      covariance: new Float64Array([variance, 0, 0, variance]),
      lastUpdate: monotonicNow(),
      lastOutput: { reward: 0, penalty: 0, total: 0 }
    }
  }

  function ensureRewardKalmanFilter(context) {
    if (!context) {
      return createRewardKalmanFilterState()
    }
    const filter = context.rewardKalman
    if (
      !filter ||
      !(filter.state instanceof Float64Array) ||
      filter.state.length !== 2 ||
      !(filter.covariance instanceof Float64Array) ||
      filter.covariance.length !== 4
    ) {
      context.rewardKalman = createRewardKalmanFilterState()
    }
    return context.rewardKalman
  }

  function applyRewardKalmanFilter(context, rewardValue, penaltyValue) {
    const rewardMeasurement = Number.isFinite(Number(rewardValue)) && Number(rewardValue) > 0
      ? Number(rewardValue)
      : 0
    const penaltyMeasurement = Number.isFinite(Number(penaltyValue)) && Number(penaltyValue) > 0
      ? Number(penaltyValue)
      : 0
    const filter = ensureRewardKalmanFilter(context)
    if (!filter || !(filter.state instanceof Float64Array) || filter.state.length !== 2) {
      return {
        reward: rewardMeasurement,
        penalty: penaltyMeasurement,
        total: rewardMeasurement - penaltyMeasurement
      }
    }

    const state = filter.state
    const covariance = filter.covariance
    const now = monotonicNow()
    const last = Number.isFinite(filter.lastUpdate) ? filter.lastUpdate : now
    const elapsedSeconds = Math.max(1, (now - last) / 1000)
    filter.lastUpdate = now

    const processNoiseValue = Math.max(minVariance, processNoise * elapsedSeconds)
    const measurementBase = Math.max(minVariance, measurementNoise)
    const measurementGrowthValue = Math.max(0, measurementGrowth)
    const rewardNoise = measurementBase + measurementGrowthValue * rewardMeasurement
    const penaltyNoise = measurementBase + measurementGrowthValue * penaltyMeasurement

    let p00 = covariance[0] + processNoiseValue
    let p01 = covariance[1]
    let p10 = covariance[2]
    let p11 = covariance[3] + processNoiseValue

    let s00 = p00 + rewardNoise
    let s01 = p01
    let s10 = p10
    let s11 = p11 + penaltyNoise
    let det = s00 * s11 - s01 * s10

    if (!Number.isFinite(det) || Math.abs(det) < minDeterminant) {
      p01 = 0
      p10 = 0
      s01 = 0
      s10 = 0
      s00 = Math.max(minVariance, s00)
      s11 = Math.max(minVariance, s11)
      det = s00 * s11
    }

    const invDet = det !== 0 ? 1 / det : 0
    const inv00 = s11 * invDet
    const inv01 = -s01 * invDet
    const inv10 = -s10 * invDet
    const inv11 = s00 * invDet

    const k00 = p00 * inv00 + p01 * inv10
    const k01 = p00 * inv01 + p01 * inv11
    const k10 = p10 * inv00 + p11 * inv10
    const k11 = p10 * inv01 + p11 * inv11

    const residual0 = rewardMeasurement - state[0]
    const residual1 = penaltyMeasurement - state[1]

    const adjustment0 = k00 * residual0 + k01 * residual1
    const adjustment1 = k10 * residual0 + k11 * residual1

    const updatedReward = state[0] + adjustment0
    const updatedPenalty = state[1] + adjustment1

    state[0] = Number.isFinite(updatedReward) ? updatedReward : rewardMeasurement
    state[1] = Number.isFinite(updatedPenalty) ? updatedPenalty : penaltyMeasurement

    const i00 = 1 - k00
    const i01 = -k01
    const i10 = -k10
    const i11 = 1 - k11

    let newP00 = i00 * p00 + i01 * p10
    let newP01 = i00 * p01 + i01 * p11
    let newP10 = i10 * p00 + i11 * p10
    let newP11 = i10 * p01 + i11 * p11

    const symOffDiag = (Number.isFinite(newP01) && Number.isFinite(newP10))
      ? (newP01 + newP10) / 2
      : 0
    covariance[0] = clampKalmanVariance(Number.isFinite(newP00) ? newP00 : initialVariance)
    covariance[3] = clampKalmanVariance(Number.isFinite(newP11) ? newP11 : initialVariance)
    const offDiag = clampKalmanVariance(symOffDiag)
    covariance[1] = offDiag
    covariance[2] = offDiag

    const filteredReward = limitPositive(state[0], maxRewardMagnitude)
    const filteredPenalty = limitPositive(state[1], maxRewardMagnitude)
    const filteredTotal = clampReward(filteredReward - filteredPenalty)

    filter.lastOutput = {
      reward: filteredReward,
      penalty: filteredPenalty,
      total: filteredTotal
    }

    return filter.lastOutput
  }

  return {
    createRewardKalmanFilterState,
    ensureRewardKalmanFilter,
    applyRewardKalmanFilter
  }
}
