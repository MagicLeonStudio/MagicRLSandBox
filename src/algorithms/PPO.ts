import { NeuralNetwork } from "./NeuralNetwork";

interface Trajectory {
  state: number[];
  action: number;
  reward: number;
  oldLogProb: number;
  oldProbs: number[];
}

export class PPO {
  actor: NeuralNetwork;
  critic: NeuralNetwork;
  learningRate: number;
  gamma: number;
  clipEpsilon: number;
  entropyCoef: number;
  maxGradNorm: number;
  actionSize: number;
  stateSize: number;

  trajectories: Trajectory[];

  constructor(
    stateSize: number,
    actionSize: number,
    params: {
      learningRate: number;
      gamma: number;
      clipEpsilon: number;
      entropyCoef: number;
      maxGradNorm?: number;
    }
  ) {
    this.stateSize = stateSize;
    this.actionSize = actionSize;
    this.actor = new NeuralNetwork([stateSize, 64, 64, actionSize]);
    this.critic = new NeuralNetwork([stateSize, 64, 64, 1]);
    this.learningRate = params.learningRate;
    this.gamma = params.gamma;
    this.clipEpsilon = params.clipEpsilon;
    this.entropyCoef = params.entropyCoef;
    this.maxGradNorm = params.maxGradNorm ?? 0.5;
    this.trajectories = [];
  }

  private getLogits(state: number[]): number[] {
    return this.actor.forward(state);
  }

  private getValue(state: number[]): number {
    return this.critic.forward(state)[0];
  }

  private logProb(logits: number[], action: number): number {
    // Safe softmax with log-probability
    const maxLogit = Math.max(...logits);
    // Clamp logit differences to prevent overflow
    const expLogits = logits.map((v) =>
      Math.exp(Math.max(-50, Math.min(v - maxLogit, 50)))
    );
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    // Prevent division by zero
    const safeSum = sumExp < 1e-10 ? 1e-10 : sumExp;
    const probs = expLogits.map((v) => v / safeSum);
    // Clamp probability to avoid log(0)
    const p = Math.max(1e-10, Math.min(1 - 1e-10, probs[action]));
    return Math.log(p);
  }

  private getProbs(logits: number[]): number[] {
    const maxLogit = Math.max(...logits);
    const expLogits = logits.map((v) =>
      Math.exp(Math.max(-50, Math.min(v - maxLogit, 50)))
    );
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    const safeSum = sumExp < 1e-10 ? 1e-10 : sumExp;
    const probs = expLogits.map((v) => v / safeSum);
    // Normalize to ensure they sum to 1
    const sum = probs.reduce((a, b) => a + b, 0);
    return probs.map((p) => p / (sum < 1e-10 ? 1 : sum));
  }

  selectAction(state: number[], _epsilonOverride?: number): number {
    const logits = this.getLogits(state);
    const maxLogit = Math.max(...logits);
    const expLogits = logits.map((v) =>
      Math.exp(Math.max(-50, Math.min(v - maxLogit, 50)))
    );
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    const safeSum = sumExp < 1e-10 ? 1e-10 : sumExp;
    const probs = expLogits.map((v) => v / safeSum);

    let r = Math.random();
    for (let i = 0; i < probs.length; i++) {
      r -= probs[i];
      if (r <= 0) return i;
    }
    return this.actionSize - 1;
  }

  private computeEntropy(probs: number[]): number {
    let entropy = 0;
    for (let j = 0; j < this.actionSize; j++) {
      if (probs[j] > 1e-10) {
        entropy -= probs[j] * Math.log(probs[j]);
      }
    }
    return entropy;
  }

  private clipGradients(grad: number[]): number[] {
    // Compute L2 norm
    let norm = 0;
    for (let j = 0; j < grad.length; j++) {
      norm += grad[j] * grad[j];
    }
    norm = Math.sqrt(norm);

    if (norm > this.maxGradNorm) {
      const scale = this.maxGradNorm / norm;
      return grad.map((g) => g * scale);
    }
    return grad;
  }

  train(
    state: number[],
    action: number,
    reward: number,
    _nextState: number[],
    done: boolean
  ): number {
    const logits = this.getLogits(state);
    const oldLogProb = this.logProb(logits, action);
    const oldProbs = this.getProbs(logits);

    // Defensive: reject NaN log probs
    if (!Number.isFinite(oldLogProb)) {
      return 0;
    }

    this.trajectories.push({
      state: [...state],
      action,
      reward,
      oldLogProb,
      oldProbs,
    });

    if (!done) return 0;

    // Compute returns
    const returns: number[] = [];
    let G = 0;
    for (let i = this.trajectories.length - 1; i >= 0; i--) {
      G = this.trajectories[i].reward + this.gamma * G;
      returns.unshift(G);
    }

    // Defensive: skip if returns contain NaN/Infinity
    if (returns.some((v) => !Number.isFinite(v))) {
      this.trajectories = [];
      return 0;
    }

    // Normalize returns (with GAE-like scaling)
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance =
      returns.reduce((a, b) => a + (b - mean) * (b - mean), 0) /
      returns.length;
    const std = Math.sqrt(variance + 1e-8);

    // Defensive: if std is too small, use unnormalized returns
    let normalizedReturns: number[];
    if (std < 1e-6) {
      normalizedReturns = returns.map(() => 0); // All advantages ≈ 0
    } else {
      normalizedReturns = returns.map((v) => (v - mean) / std);
    }

    let totalLoss = 0;

    // PPO: multiple epochs of updates
    for (let epoch = 0; epoch < 4; epoch++) {
      for (let i = 0; i < this.trajectories.length; i++) {
        const t = this.trajectories[i];

        // Critic update first (before actor forward overwrites critic's activations)
        const value = this.getValue(t.state);
        const criticTarget = [normalizedReturns[i]];
        const criticLoss = this.critic.backward(
          criticTarget,
          this.learningRate
        );

        // Compute advantage
        const advantage = normalizedReturns[i] - value;

        // Actor: compute new log prob and ratio
        const newLogits = this.getLogits(t.state);
        const newLogProb = this.logProb(newLogits, t.action);
        const newProbs = this.getProbs(newLogits);

        // Defensive: skip if NaN detected
        if (!Number.isFinite(newLogProb)) continue;

        // Ratio with clipping
        const ratio = Math.exp(
          Math.max(-10, Math.min(newLogProb - t.oldLogProb, 10))
        );

        // PPO clipped surrogate objective gradient
        let useClipped = false;
        if (advantage > 0 && ratio > 1 + this.clipEpsilon)
          useClipped = true;
        if (advantage < 0 && ratio < 1 - this.clipEpsilon)
          useClipped = true;

        const actorGrad = new Array(this.actionSize).fill(0);

        if (!useClipped) {
          // Standard policy gradient: -A * ∇log π(a)
          // ∇log π(a) / ∇z_j = (1{j==a} - π(j))
          for (let j = 0; j < this.actionSize; j++) {
            actorGrad[j] =
              -advantage * ((j === t.action ? 1 : 0) - newProbs[j]);
          }
        }

        // Entropy bonus: maximize entropy H = -Σ p_j log p_j
        // dH/dz_j = p_j * (H + log p_j)
        const entropy = this.computeEntropy(newProbs);
        for (let j = 0; j < this.actionSize; j++) {
          if (newProbs[j] > 1e-10) {
            const entropyGrad =
              newProbs[j] * (entropy + Math.log(newProbs[j]));
            actorGrad[j] += this.entropyCoef * entropyGrad;
          }
        }

        // Clip gradients before applying
        const clippedGrad = this.clipGradients(actorGrad);

        const actorGradNorm = this.actor.backwardOutputGradient(
          clippedGrad,
          this.learningRate
        );

        totalLoss += criticLoss + Math.abs(actorGradNorm);
      }
    }

    const avgLoss =
      this.trajectories.length > 0
        ? totalLoss / (this.trajectories.length * 4)
        : 0;

    this.trajectories = [];

    return Number.isFinite(avgLoss) ? avgLoss : 0;
  }

  getPolicy(): number[] {
    if (this.trajectories.length === 0) {
      return new Array(this.actionSize).fill(1 / this.actionSize);
    }
    const logits = this.getLogits(
      this.trajectories[this.trajectories.length - 1].state
    );
    return this.getProbs(logits);
  }

  reset(): void {
    this.actor = new NeuralNetwork([this.stateSize, 64, 64, this.actionSize]);
    this.critic = new NeuralNetwork([this.stateSize, 64, 64, 1]);
    this.trajectories = [];
  }
}
