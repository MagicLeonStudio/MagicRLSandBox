import { NeuralNetwork } from "@/algorithms/NeuralNetwork";
import { mulberry32 } from "@/lib/rng";
import type { CartPoleAction } from "@/engines/CartPoleEngine";
import type { CartPoleEngine } from "@/engines/CartPoleEngine";

export interface CartPoleExperience {
  s: number[];
  a: CartPoleAction;
  r: number;
  s2: number[];
  done: boolean;
}

export interface CartPoleTrainInfo {
  target: number;
  prediction: number;
  tdError: number;
  maxNext: number;
  loss: number;
}

export interface CartPoleEvalPoint {
  step: number;
  successRate: number;
  avgSteps: number;
}

export interface CartPoleDQNParams {
  seed: number;
  useReplay: boolean;
  useTargetNet: boolean;
  lr?: number;
  gamma?: number;
  batchSize?: number;
  bufferSize?: number;
  targetUpdate?: number;
  epsilonStart?: number;
  epsilonMin?: number;
  epsilonDecay?: number;
}

/** Seeded DQN for the CartPole stability stress test. */
export class CartPoleDQN {
  online: NeuralNetwork;
  target: NeuralNetwork;
  params: Required<Omit<CartPoleDQNParams, "seed">> & { seed: number };
  buffer: CartPoleExperience[] = [];
  lastTrain: CartPoleTrainInfo | null = null;
  evalHistory: CartPoleEvalPoint[] = [];
  stepCount = 0;
  lastSyncStep = 0;
  epsilon: number;
  private rng: () => number;

  constructor(params: CartPoleDQNParams) {
    this.params = {
      seed: params.seed,
      useReplay: params.useReplay,
      useTargetNet: params.useTargetNet,
      lr: params.lr ?? 0.001,
      gamma: params.gamma ?? 0.99,
      batchSize: params.batchSize ?? 32,
      bufferSize: params.bufferSize ?? 10000,
      targetUpdate: params.targetUpdate ?? 500,
      epsilonStart: params.epsilonStart ?? 1,
      epsilonMin: params.epsilonMin ?? 0.05,
      epsilonDecay: params.epsilonDecay ?? 0.999,
    };
    this.online = new NeuralNetwork([4, 64, 64, 2], params.seed);
    this.target = new NeuralNetwork([4, 64, 64, 2], params.seed);
    this.target.setWeights(this.online.getWeights().weights, this.online.getWeights().biases);
    this.rng = mulberry32(params.seed + 100003);
    this.epsilon = this.params.epsilonStart;
  }

  selectAction(state: number[], epsilon: number): CartPoleAction {
    if (this.rng() < epsilon) return this.rng() < 0.5 ? 0 : 1;
    const q = this.online.forward(state);
    return (q[1] > q[0] ? 1 : 0) as CartPoleAction;
  }

  trainStep(s: number[], a: CartPoleAction, r: number, s2: number[], done: boolean): number {
    this.stepCount++;
    this.epsilon = Math.max(this.params.epsilonMin, this.epsilon * this.params.epsilonDecay);
    this.buffer.push({ s: [...s], a, r, s2: [...s2], done });
    if (this.buffer.length > this.params.bufferSize) this.buffer.shift();

    if (this.params.useTargetNet && this.stepCount % this.params.targetUpdate === 0) {
      const weights = this.online.getWeights();
      this.target.setWeights(weights.weights, weights.biases);
      this.lastSyncStep = this.stepCount;
    }

    if (!this.params.useReplay) return this.update({ s, a, r, s2, done }, this.params.lr);
    if (this.buffer.length < this.params.batchSize) return 0;
    let loss = 0;
    for (let i = 0; i < this.params.batchSize; i++) {
      const exp = this.buffer[Math.floor(this.rng() * this.buffer.length)];
      loss += this.update(exp, this.params.lr / this.params.batchSize);
    }
    return loss / this.params.batchSize;
  }

  private update(exp: CartPoleExperience, lr: number): number {
    const nextQ = (this.params.useTargetNet ? this.target : this.online).forward(exp.s2);
    const maxNext = exp.done ? 0 : Math.max(...nextQ);
    const q = this.online.forward(exp.s);
    const targetQ = [...q];
    const y = exp.r + this.params.gamma * maxNext;
    targetQ[exp.a] = y;
    const prediction = q[exp.a];
    const loss = this.online.backward(targetQ, lr);
    this.lastTrain = { target: y, prediction, tdError: y - prediction, maxNext, loss };
    return loss;
  }

  evaluate(env: CartPoleEngine, episodes = 10, maxSteps = 200): CartPoleEvalPoint {
    let success = 0;
    let totalSteps = 0;
    for (let episode = 0; episode < episodes; episode++) {
      env.reset();
      let steps = 0;
      for (let t = 0; t < maxSteps; t++) {
        const action = this.selectAction(env.getObservation(), 0);
        const { done } = env.step(action);
        steps++;
        if (done) break;
      }
      if (steps >= maxSteps) success++;
      totalSteps += steps;
    }
    const point = { step: this.stepCount, successRate: success / episodes, avgSteps: totalSteps / episodes };
    this.evalHistory.push(point);
    return point;
  }
}
