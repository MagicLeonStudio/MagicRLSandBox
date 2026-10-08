/**
 * CliffWalk —— Sutton & Barto 例 6.6 悬崖行走环境 + 表格型 Q-learning / SARSA
 * 网格 4×12：起点左下 (3,0)，终点右下 (3,11)，下排中间 10 格为悬崖。
 * 每步 -1，掉落悬崖 -100 并回到起点。S&B 原设定 gamma = 1。
 */

export const CW_ROWS = 4;
export const CW_COLS = 12;
export const CW_START = 3 * CW_COLS + 0;   // (3,0)
export const CW_GOAL = 3 * CW_COLS + 11;   // (3,11)
export const CW_CLIFF = new Set(Array.from({ length: 10 }, (_, i) => 3 * CW_COLS + 1 + i));

export function cwActions(s: number): number[] {
  if (s === CW_GOAL) return [];
  return [0, 1, 2, 3]; // 上 下 左 右
}

export function cwNext(s: number, a: number): number {
  const r = Math.floor(s / CW_COLS);
  const c = s % CW_COLS;
  let nr = r, nc = c;
  if (a === 0) nr = Math.max(0, r - 1);
  else if (a === 1) nr = Math.min(CW_ROWS - 1, r + 1);
  else if (a === 2) nc = Math.max(0, c - 1);
  else if (a === 3) nc = Math.min(CW_COLS - 1, c + 1);
  return nr * CW_COLS + nc;
}

/** 确定性转移 + 悬崖掉落回起点；返回 {ns, reward, fell} */
export function cwStep(s: number, a: number): { ns: number; reward: number; fell: boolean } {
  const ns = cwNext(s, a);
  if (CW_CLIFF.has(ns)) return { ns: CW_START, reward: -100, fell: true };
  if (ns === CW_GOAL) return { ns, reward: -1, fell: false };
  return { ns, reward: -1, fell: false };
}

/** ε-greedy 动作选择 */
export function epsGreedy(q: number[], s: number, eps: number, rng: () => number): number {
  const acts = cwActions(s);
  if (rng() < eps) return acts[Math.floor(rng() * acts.length)];
  let best = acts[0];
  for (const a of acts) if (q[s * 4 + a] > q[s * 4 + best]) best = a;
  return best;
}

/** 表格型 Q-learning（off-policy：目标用 max_a' Q(s',a')） */
export class TabularQLearning {
  q: number[];
  private alpha: number;
  constructor(alpha: number) {
    this.alpha = alpha;
    this.q = new Array(CW_ROWS * CW_COLS * 4).fill(0);
  }
  observeStep(s: number, a: number, r: number, s2: number, done: boolean, gamma: number): void {
    const acts2 = cwActions(s2);
    let maxQ = 0;
    if (acts2.length > 0) {
      maxQ = Math.max(...acts2.map(x => this.q[s2 * 4 + x]));
    }
    const idx = s * 4 + a;
    this.q[idx] += this.alpha * (r + (done ? 0 : gamma * maxQ) - this.q[idx]);
  }
  greedyPolicy(): number[] {
    const pi = new Array(CW_ROWS * CW_COLS).fill(-1);
    for (let s = 0; s < pi.length; s++) {
      const acts = cwActions(s);
      if (acts.length === 0) continue;
      let best = acts[0];
      for (const a of acts) if (this.q[s * 4 + a] > this.q[s * 4 + best]) best = a;
      pi[s] = best;
    }
    return pi;
  }
}

/** 表格型 SARSA（on-policy：目标用实际执行的 a' 的 Q(s',a')） */
export class TabularSARSA {
  q: number[];
  private alpha: number;
  constructor(alpha: number) {
    this.alpha = alpha;
    this.q = new Array(CW_ROWS * CW_COLS * 4).fill(0);
  }
  observeStep(s: number, a: number, r: number, s2: number, a2: number | null, done: boolean, gamma: number): void {
    const idx = s * 4 + a;
    const nextQ = done || a2 === null ? 0 : this.q[s2 * 4 + a2];
    this.q[idx] += this.alpha * (r + (done ? 0 : gamma * nextQ) - this.q[idx]);
  }
  greedyPolicy(): number[] {
    const pi = new Array(CW_ROWS * CW_COLS).fill(-1);
    for (let s = 0; s < pi.length; s++) {
      const acts = cwActions(s);
      if (acts.length === 0) continue;
      let best = acts[0];
      for (const a of acts) if (this.q[s * 4 + a] > this.q[s * 4 + best]) best = a;
      pi[s] = best;
    }
    return pi;
  }
}

/** 跑一个 episode（SARSA 需要先选 a'；Q-learning 逐步即可）。返回 episode 总奖励。 */
export function runEpisodeQLearning(algo: TabularQLearning, eps: number, gamma: number, rng: () => number): number {
  let s = CW_START, total = 0, steps = 0;
  while (s !== CW_GOAL && steps < 500) {
    const a = epsGreedy(algo.q, s, eps, rng);
    const { ns, reward } = cwStep(s, a);
    algo.observeStep(s, a, reward, ns, ns === CW_GOAL, gamma);
    total += reward;
    s = ns;
    steps++;
  }
  return total;
}

export function runEpisodeSARSA(algo: TabularSARSA, eps: number, gamma: number, rng: () => number): number {
  let s = CW_START, total = 0, steps = 0;
  let a = epsGreedy(algo.q, s, eps, rng);
  while (s !== CW_GOAL && steps < 500) {
    const { ns, reward } = cwStep(s, a);
    const done = ns === CW_GOAL;
    const a2 = done ? null : epsGreedy(algo.q, ns, eps, rng);
    algo.observeStep(s, a, reward, ns, a2, done, gamma);
    total += reward;
    s = ns;
    a = a2 as number;
    steps++;
  }
  return total;
}
