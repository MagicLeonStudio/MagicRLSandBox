import { GridWorldModel } from "./GridWorldModel";

/** 可播种的 LCG 随机数发生器：同一 seed 产生同一轨迹序列，保证 MC 与 TD 公平对比 */
export function makeRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/** 固定策略：均匀随机。评估对象是这个"什么都不懂"的策略的 V^pi */
export function uniformPolicy(model: GridWorldModel, s: number, rng: () => number): number {
  const acts = model.actions(s);
  return acts[Math.floor(rng() * acts.length)];
}

export interface Trajectory {
  states: number[];   // s_0, s_1, ..., s_T（含终点）
  actions: number[];  // a_0, ..., a_{T-1}
  rewards: number[];  // r_1, ..., r_T
}

export function sampleEpisode(
  model: GridWorldModel,
  rng: () => number,
  maxSteps = 200
): Trajectory {
  const states: number[] = [0];
  const actions: number[] = [];
  const rewards: number[] = [];
  let s = 0;
  for (let t = 0; t < maxSteps && !model.isTerminal(s); t++) {
    const a = uniformPolicy(model, s, rng);
    const ns = model.nextState(s, a);
    actions.push(a);
    rewards.push(model.reward(s, a, ns));
    s = ns;
    states.push(s);
  }
  return { states, actions, rewards };
}

/** 折扣回报 G_t = sum_k gamma^k r_{t+k+1}（在整条轨迹算好后反查） */
export function discountedReturns(traj: Trajectory, gamma: number): number[] {
  const T = traj.rewards.length;
  const G = new Array<number>(T).fill(0);
  let g = 0;
  for (let t = T - 1; t >= 0; t--) {
    g = traj.rewards[t] + gamma * g;
    G[t] = g;
  }
  return G;
}

/** 首次访问蒙特卡洛：episode 结束后，用实际回报 G 更新每个状态的首访估计 */
export class FirstVisitMC {
  v: number[];
  private counts: number[];
  constructor(model: GridWorldModel) {
    this.v = new Array(model.stateCount).fill(0);
    this.counts = new Array(model.stateCount).fill(0);
  }
  observeEpisode(traj: Trajectory, gamma: number): number[] {
    const G = discountedReturns(traj, gamma);
    const seen = new Set<number>();
    const updated: number[] = [];
    for (let t = 0; t < traj.states.length - 1; t++) {
      const s = traj.states[t];
      if (seen.has(s)) continue;
      seen.add(s);
      this.counts[s] += 1;
      this.v[s] += (G[t] - this.v[s]) / this.counts[s]; // 增量均值（无偏）
      updated.push(s);
    }
    return updated;
  }
}

/** TD(0)：每走一步立即 bootstrap 更新。TD error delta = r + gamma V(s') - V(s) */
export class TDZero {
  v: number[];
  private alpha: number;
  constructor(model: GridWorldModel, alpha: number) {
    this.alpha = alpha;
    this.v = new Array(model.stateCount).fill(0);
  }
  observeStep(s: number, r: number, s2: number, done: boolean, gamma: number): number {
    const target = r + (done ? 0 : gamma * this.v[s2]);
    const delta = target - this.v[s];
    this.v[s] += this.alpha * delta;
    return delta;
  }
}

/** TD(lambda)：累积迹（accumulating eligibility traces），lambda=0 退化为 TD(0)，lambda=1 近似 MC */
export class TDLambda {
  v: number[];
  traces: number[];
  private alpha: number;
  private lambda: number;
  constructor(model: GridWorldModel, alpha: number, lambda: number) {
    this.alpha = alpha;
    this.lambda = lambda;
    this.v = new Array(model.stateCount).fill(0);
    this.traces = new Array(model.stateCount).fill(0);
  }
  observeStep(s: number, r: number, s2: number, done: boolean, gamma: number): number {
    this.traces[s] += 1;
    const target = r + (done ? 0 : gamma * this.v[s2]);
    const delta = target - this.v[s];
    for (let i = 0; i < this.v.length; i++) {
      this.v[i] += this.alpha * delta * this.traces[i];
      this.traces[i] *= gamma * this.lambda;
    }
    if (done) this.traces.fill(0);
    return delta;
  }
}

/** 真值：固定策略的 Bellman 期望算子迭代收敛到 V^pi（DP 参照系） */
export function evaluatePolicyExact(
  model: GridWorldModel,
  gamma: number,
  theta = 1e-10
): number[] {
  let v = new Array<number>(model.stateCount).fill(0);
  let delta = Infinity;
  let guard = 0;
  while (delta > theta && guard < 100000) {
    delta = 0;
    const next = [...v];
    for (let s = 0; s < model.stateCount; s++) {
      const acts = model.actions(s);
      if (acts.length === 0) { next[s] = 0; continue; }
      // 均匀随机策略的期望
      let q = 0;
      for (const a of acts) {
        const ns = model.nextState(s, a);
        q += model.reward(s, a, ns) + gamma * v[ns];
      }
      next[s] = q / acts.length;
      delta = Math.max(delta, Math.abs(next[s] - v[s]));
    }
    v = next;
    guard++;
  }
  return v;
}
