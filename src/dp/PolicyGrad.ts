import { GridWorldModel } from "./GridWorldModel";
import { discountedReturns, type Trajectory } from "./MCTD";

/**
 * SoftmaxPolicy —— 参数化策略 π_θ(a|s) = exp(θ_sa) / Σ_a' exp(θ_sa')
 * REINFORCE 的直接优化对象：不再估计价值，而是沿着 ∇log π · G 推高好动作的概率。
 */
export class SoftmaxPolicy {
  theta: number[]; // 长度 stateCount * actionSize

  constructor(model: GridWorldModel, actionSize = 4) {

    this.theta = new Array(model.stateCount * actionSize).fill(0);
  }
  probs(s: number): number[] {
    const acts = 4;
    const logits = Array.from({ length: acts }, (_, a) => this.theta[s * acts + a]);
    const maxL = Math.max(...logits);
    const exps = logits.map(l => Math.exp(l - maxL));
    const sum = exps.reduce((x, y) => x + y, 0);
    return exps.map(e => e / sum);
  }
  sample(s: number, rng: () => number): number {
    const p = this.probs(s);
    let r = rng();
    for (let a = 0; a < p.length; a++) {
      r -= p[a];
      if (r <= 0) return a;
    }
    return p.length - 1;
  }
  /** softmax 的 ∇log π(a|s)：one-hot(a) − π(·|s) */
  gradLogPi(s: number, a: number): number[] {
    const p = this.probs(s);
    return p.map((pk, k) => (k === a ? 1 - pk : -pk));
  }
  greedyAction(s: number): number {
    const p = this.probs(s);
    let best = 0;
    for (let a = 1; a < p.length; a++) if (p[a] > p[best]) best = a;
    return best;
  }
  greedyPolicy(model: GridWorldModel): number[] {
    const pi = new Array(model.stateCount).fill(-1);
    for (let s = 0; s < pi.length; s++) {
      if (model.actions(s).length === 0) continue;
      pi[s] = this.greedyAction(s);
    }
    return pi;
  }
}

/** 用给定策略采样一个 episode（上限 maxSteps 步） */
export function sampleEpisodeWithPolicy(
  model: GridWorldModel,
  policy: SoftmaxPolicy,
  rng: () => number,
  maxSteps = 200
): Trajectory {
  const states: number[] = [0];
  const actions: number[] = [];
  const rewards: number[] = [];
  let s = 0;
  for (let t = 0; t < maxSteps && !model.isTerminal(s); t++) {
    const a = policy.sample(s, rng);
    const ns = model.nextState(s, a);
    actions.push(a);
    rewards.push(model.reward(s, a, ns));
    s = ns;
    states.push(s);
  }
  return { states, actions, rewards };
}

/**
 * REINFORCE：episode 结束后，沿 ∇log π(A_t|S_t) · (G_t − b) 更新 θ。
 * baseline b 为移动平均回报（不改变无偏性，降低方差）。
 */
export class REINFORCE {
  policy: SoftmaxPolicy;
  baseline: number = 0;
  private baselineN = 0;
  private alpha: number;
  private useBaseline: boolean;
  constructor(policy: SoftmaxPolicy, alpha: number, useBaseline: boolean) {
    this.policy = policy;
    this.alpha = alpha;
    this.useBaseline = useBaseline;
  }

  observeEpisode(_model: GridWorldModel, traj: Trajectory, gamma: number): { gradNorm: number; steps: number } {
    const G = discountedReturns(traj, gamma);
    const T = traj.rewards.length;
    const totalReturn = traj.rewards.reduce((x, y) => x + y, 0);

    // 更新 baseline（episode 均值的增量形式）
    this.baselineN += 1;
    this.baseline += (totalReturn - this.baseline) / this.baselineN;
    const b = this.useBaseline ? this.baseline : 0;

    let gradNorm = 0;
    const acts = 4;
    for (let t = 0; t < T; t++) {
      const s = traj.states[t];
      const a = traj.actions[t];
      const adv = G[t] - b;
      const g = this.policy.gradLogPi(s, a);
      for (let k = 0; k < acts; k++) {
        const upd = this.alpha * adv * g[k];
        this.policy.theta[s * acts + k] += upd;
        gradNorm += upd * upd;
      }
    }
    return { gradNorm: Math.sqrt(gradNorm), steps: T };
  }
}
