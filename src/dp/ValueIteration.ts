import { GridWorldModel } from "./GridWorldModel";

export interface SweepResult {
  v: number[];
  delta: number;
  sweep: number;
  done: boolean;
}

/**
 * Value Iteration（价值迭代）
 * V_{k+1}(s) = max_a Σ_s' P(s'|s,a)[r + γ V_k(s')]
 * 每轮全量扫描（sweep）所有状态，直到 max|V_{k+1}-V_k| < theta。
 */
export class ValueIteration {
  v: number[];
  sweeps = 0;
  deltas: number[] = [];
  snapshots: number[][] = [];
  private model: GridWorldModel;
  private gamma: number;
  private theta: number;

  constructor(model: GridWorldModel, gamma = 0.99, theta = 1e-6) {
    this.model = model;
    this.gamma = gamma;
    this.theta = theta;
    this.v = new Array(model.stateCount).fill(0);
    this.snapshots.push([...this.v]);
  }

  /** 执行一轮 Bellman 最优算子扫描，返回本轮结果 */
  sweep(): SweepResult {
    const next = [...this.v];
    let delta = 0;
    for (let s = 0; s < this.model.stateCount; s++) {
      const acts = this.model.actions(s);
      if (acts.length === 0) { next[s] = 0; continue; }
      let best = -Infinity;
      for (const a of acts) {
        const ns = this.model.nextState(s, a);
        const q = this.model.reward(s, a, ns) + this.gamma * this.v[ns];
        if (q > best) best = q;
      }
      delta = Math.max(delta, Math.abs(best - this.v[s]));
      next[s] = best;
    }
    this.v = next;
    this.sweeps += 1;
    this.deltas.push(delta);
    this.snapshots.push([...this.v]);
    return { v: [...this.v], delta, sweep: this.sweeps, done: delta < this.theta };
  }

  /** 从当前 V 提取确定性贪心策略（动作编号数组，无效状态为 -1） */
  greedyPolicy(): number[] {
    const n = this.model.stateCount;
    const pi = new Array(n).fill(-1);
    for (let s = 0; s < n; s++) {
      const acts = this.model.actions(s);
      if (acts.length === 0) continue;
      let best = -Infinity;
      for (const a of acts) {
        const ns = this.model.nextState(s, a);
        const q = this.model.reward(s, a, ns) + this.gamma * this.v[ns];
        if (q > best) { best = q; pi[s] = a; }
      }
    }
    return pi;
  }
}
