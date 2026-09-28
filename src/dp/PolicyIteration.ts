import { GridWorldModel } from "./GridWorldModel";

export interface PIResult {
  v: number[];
  policy: number[];
  phase: "eval" | "improve";
  evalDelta: number;
  stable: boolean;
  iteration: number;
  done: boolean;
}

/**
 * Policy Iteration（策略迭代）
 * 评估：V ← V^π（迭代至 delta < thetaEval）；改进：π(s) ← argmax_a Q(s,a)。
 * step() 完成一次完整的"评估 + 改进"，策略稳定时 done。
 */
export class PolicyIteration {
  v: number[];
  policy: number[];
  iterations = 0;
  history: PIResult[] = [];
  private model: GridWorldModel;
  private gamma: number;
  private thetaEval: number;

  constructor(model: GridWorldModel, gamma = 0.99, thetaEval = 1e-4) {
    this.model = model;
    this.gamma = gamma;
    this.thetaEval = thetaEval;
    this.v = new Array(model.stateCount).fill(0);
    // 初始策略：第一个合法动作
    this.policy = Array.from({ length: model.stateCount }, (_, s) => {
      const acts = model.actions(s);
      return acts.length ? acts[0] : -1;
    });
  }

  /** 策略评估：迭代应用 Bellman 期望算子直到收敛 */
  private evaluate(): number {
    let delta = Infinity;
    let guard = 0;
    while (delta > this.thetaEval && guard < 10000) {
      delta = 0;
      const next = [...this.v];
      for (let s = 0; s < this.model.stateCount; s++) {
        const a = this.policy[s];
        if (a < 0) { next[s] = 0; continue; }
        const ns = this.model.nextState(s, a);
        const q = this.model.reward(s, a, ns) + this.gamma * this.v[ns];
        delta = Math.max(delta, Math.abs(q - this.v[s]));
        next[s] = q;
      }
      this.v = next;
      guard++;
    }
    return delta;
  }

  /** 策略改进：贪心化；返回策略是否稳定 */
  private improve(): boolean {
    let stable = true;
    for (let s = 0; s < this.model.stateCount; s++) {
      const acts = this.model.actions(s);
      if (acts.length === 0) continue;
      const old = this.policy[s];
      let best = -Infinity;
      for (const a of acts) {
        const ns = this.model.nextState(s, a);
        const q = this.model.reward(s, a, ns) + this.gamma * this.v[ns];
        if (q > best) { best = q; this.policy[s] = a; }
      }
      if (this.policy[s] !== old) stable = false;
    }
    return stable;
  }

  /** 一次完整迭代：评估到收敛 + 一次贪心改进 */
  step(): PIResult {
    const evalDelta = this.evaluate();
    const stable = this.improve();
    this.iterations += 1;
    const result: PIResult = {
      v: [...this.v],
      policy: [...this.policy],
      phase: "improve",
      evalDelta,
      stable,
      iteration: this.iterations,
      done: stable,
    };
    this.history.push(result);
    return result;
  }
}
