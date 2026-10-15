import { NeuralNetwork } from "@/algorithms/NeuralNetwork";

/**
 * DQNAblation —— DQN 消融实验：可分别关掉经验回放与目标网络。
 * 复刻 Mnih et al. 2015 Nature 论文的消融思路：full DQN / no-replay / no-target-net。
 * 环境：连续观测 [agentX/4, agentY/4, goalX/4, goalY/4]，神经网络 [4, 64, 64, 4]。
 */

interface Experience {
  s: number[];
  a: number;
  r: number;
  s2: number[];
  done: boolean;
}

export class DQNAblation {
  params: { lr: number; gamma: number; batchSize: number; bufferSize: number; targetUpdate: number; useReplay: boolean; useTargetNet: boolean; epsilonStart?: number; epsilonMin?: number; epsilonDecay?: number; hidden?: number[] };
  online: NeuralNetwork;
  target: NeuralNetwork;
  buffer: Experience[];
  private stepCount = 0;
  epsilon: number;
  private epsilonDecay: number;

  constructor(
    stateSize: number,
    actionSize: number,
    params: {
      lr: number;
      gamma: number;
      batchSize: number;
      bufferSize: number;
      targetUpdate: number;
      useReplay: boolean;
      useTargetNet: boolean;
      epsilonStart?: number;
      epsilonMin?: number;
      epsilonDecay?: number;
      hidden?: number[];
    }
  ) {
    this.params = params;
    const hidden = params.hidden ?? [64, 64];
    this.online = new NeuralNetwork([stateSize, ...hidden, actionSize]);
    this.target = new NeuralNetwork([stateSize, ...hidden, actionSize]);
    this.syncTarget();
    this.buffer = [];
    this.epsilon = params.epsilonStart ?? 1.0;
    this.epsilonDecay = params.epsilonDecay ?? 0.995;
  }

  private syncTarget() {
    const w = this.online.getWeights();
    this.target.setWeights(w.weights, w.biases);
  }

  selectAction(s: number[], eps: number): number {
    if (Math.random() < eps) return Math.floor(Math.random() * 4);
    const q = this.online.forward(s);
    let best = 0;
    for (let i = 1; i < q.length; i++) if (q[i] > q[best]) best = i;
    return best;
  }

  trainStep(s: number[], a: number, r: number, s2: number[], done: boolean): number {
    this.stepCount++;
    this.epsilon = Math.max(this.params.epsilonMin ?? 0.01, this.epsilon * this.epsilonDecay);
    if (this.params.useTargetNet && this.stepCount % this.params.targetUpdate === 0) {
      this.syncTarget();
    }
    if (this.params.useReplay) {
      this.buffer.push({ s: [...s], a, r, s2: [...s2], done });
      if (this.buffer.length > this.params.bufferSize) this.buffer.shift();
      if (this.buffer.length < this.params.batchSize) return 0;
      return this.trainBatch();
    }
    return this.update(s, a, r, s2, done);
  }

  private trainBatch(): number {
    let totalLoss = 0;
    const wD = this.online.weights.map(l => l.map(row => new Array(row.length).fill(0)));
    const bD = this.online.biases.map(b => new Array(b.length).fill(0));
    const lrPer = this.params.lr / this.params.batchSize;

    for (let i = 0; i < this.params.batchSize; i++) {
      const exp = this.buffer[Math.floor(Math.random() * this.buffer.length)];
      totalLoss += this.accumulate(exp, wD, bD);
    }
    for (let i = 0; i < this.online.weights.length; i++)
      for (let j = 0; j < this.online.weights[i].length; j++) {
        for (let k = 0; k < this.online.weights[i][j].length; k++)
          this.online.weights[i][j][k] -= lrPer * wD[i][j][k];
        this.online.biases[i][j] -= lrPer * bD[i][j];
      }
    return totalLoss / this.params.batchSize;
  }

  private update(s: number[], a: number, r: number, s2: number[], done: boolean): number {
    const targetQ = [...this.online.forward(s)];
    const q2 = (this.params.useTargetNet ? this.target : this.online).forward(s2);
    const maxQ = done ? 0 : Math.max(...q2);
    targetQ[a] = r + this.params.gamma * maxQ;
    return this.online.backward(targetQ, this.params.lr);
  }

  private accumulate(exp: Experience, wD: number[][][], bD: number[][]): number {
    const q = this.online.forward(exp.s);
    const q2 = (this.params.useTargetNet ? this.target : this.online).forward(exp.s2);
    const maxQ = exp.done ? 0 : Math.max(...q2);
    const target = exp.r + this.params.gamma * maxQ;
    const tdErr = target - q[exp.a];
    const targetQ = [...q];
    targetQ[exp.a] = target;

    const output = this.online.activations[this.online.activations.length - 1];
    const deltas: number[][] = [];
    const od: number[] = [];
    for (let i = 0; i < output.length; i++) od.push(output[i] - targetQ[i]);
    deltas.push(od);
    for (let i = this.online.weights.length - 2; i >= 0; i--) {
      const d: number[] = [];
      for (let j = 0; j < this.online.activations[i + 1].length; j++) {
        let err = 0;
        for (let k = 0; k < deltas[deltas.length - 1].length; k++)
          err += deltas[deltas.length - 1][k] * this.online.weights[i + 1][k][j];
        d.push(err * (this.online.zs[i][j] > 0 ? 1 : 0));
      }
      deltas.push(d);
    }
    deltas.reverse();
    for (let i = 0; i < this.online.weights.length; i++)
      for (let j = 0; j < this.online.weights[i].length; j++) {
        for (let k = 0; k < this.online.weights[i][j].length; k++)
          wD[i][j][k] += deltas[i][j] * this.online.activations[i][k];
        bD[i][j] += deltas[i][j];
      }
    return 0.5 * tdErr * tdErr;
  }
}

/** GridWorld 连续观测环境（供消融实验；固定目标右下角） */
export class GWContEnv {
  private pos = 0;
  private steps = 0;
  private readonly goal = 24;
  private readonly walls = new Set([6, 7, 11]);

  reset(): number[] {
    this.pos = 0;
    this.steps = 0;
    return this.obs();
  }
  private obs(): number[] {
    const r = Math.floor(this.pos / 5), c = this.pos % 5;
    const gr = Math.floor(this.goal / 5), gc = this.goal % 5;
    return [c / 4, r / 4, gc / 4, gr / 4];
  }
  step(a: number): { s2: number[]; r: number; done: boolean } {
    const r0 = Math.floor(this.pos / 5), c0 = this.pos % 5;
    let nr = r0, nc = c0;
    if (a === 0) nr = Math.max(0, r0 - 1);
    else if (a === 1) nr = Math.min(4, r0 + 1);
    else if (a === 2) nc = Math.max(0, c0 - 1);
    else nc = Math.min(4, c0 + 1);
    const ns = nr * 5 + nc;
    this.steps++;
    if (this.walls.has(ns)) return { s2: this.obs(), r: -0.1, done: false }; // 撞障碍原地
    this.pos = ns;
    if (ns === this.goal) return { s2: this.obs(), r: 10, done: true };
    if (this.steps >= 200) return { s2: this.obs(), r: -0.1, done: true };
    return { s2: this.obs(), r: -0.1, done: false };
  }
}
