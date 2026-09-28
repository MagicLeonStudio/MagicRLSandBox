import { NeuralNetwork } from "./NeuralNetwork";

export class A3C {
  network: NeuralNetwork; // shared actor-critic
  learningRate: number;
  gamma: number;
  entropyCoef: number;
  valueCoef: number;
  maxGradNorm: number;
  actionSize: number;
  stateSize: number;

  // Episode memory
  states: number[][];
  actions: number[];
  rewards: number[];

  constructor(
    stateSize: number,
    actionSize: number,
    params: {
      learningRate: number;
      gamma: number;
      entropyCoef: number;
      valueCoef?: number;
      maxGradNorm?: number;
    }
  ) {
    this.stateSize = stateSize;
    this.actionSize = actionSize;
    // Actor head: actionSize outputs, Critic head: 1 output
    this.network = new NeuralNetwork([stateSize, 64, 64, actionSize + 1]);
    this.learningRate = params.learningRate;
    this.gamma = params.gamma;
    this.entropyCoef = params.entropyCoef;
    this.valueCoef = params.valueCoef ?? 0.5;
    this.maxGradNorm = params.maxGradNorm ?? 0.5;
    this.states = [];
    this.actions = [];
    this.rewards = [];
  }

  selectAction(state: number[], epsilonOverride?: number): number {
    const output = this.network.forward(state);
    const logits = output.slice(0, this.actionSize);

    // Add exploration noise if epsilon is provided
    if (epsilonOverride && epsilonOverride > 0) {
      for (let i = 0; i < logits.length; i++) {
        logits[i] += (Math.random() - 0.5) * epsilonOverride;
      }
    }

    // Softmax
    const maxLogit = Math.max(...logits);
    const expLogits = logits.map((v) => Math.exp(Math.min(v - maxLogit, 50)));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    const probs = expLogits.map((v) => v / sumExp);

    // Sample from policy
    let r = Math.random();
    for (let i = 0; i < probs.length; i++) {
      r -= probs[i];
      if (r <= 0) return i;
    }
    return this.actionSize - 1;
  }

  train(
    state: number[],
    action: number,
    reward: number,
    _nextState: number[],
    done: boolean
  ): number {
    this.states.push([...state]);
    this.actions.push(action);
    this.rewards.push(reward);

    if (!done) return 0;

    // Compute returns
    const returns: number[] = [];
    let G = 0;
    for (let i = this.rewards.length - 1; i >= 0; i--) {
      G = this.rewards[i] + this.gamma * G;
      returns.unshift(G);
    }

    // Normalize returns
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const std =
      Math.sqrt(
        returns.reduce((a, b) => a + (b - mean) * (b - mean), 0) /
          returns.length +
          1e-8
      );
    const normalizedReturns = returns.map((v) => (v - mean) / std);

    let totalLoss = 0;

    for (let i = 0; i < this.states.length; i++) {
      const output = this.network.forward(this.states[i]);
      const logits = output.slice(0, this.actionSize);
      const value = output[this.actionSize];

      const advantage = normalizedReturns[i] - value;

      // Compute policy probabilities
      const maxLogit = Math.max(...logits);
      const expLogits = logits.map((v) => Math.exp(Math.min(v - maxLogit, 50)));
      const sumExp = expLogits.reduce((a, b) => a + b, 0);
      const safeSum = sumExp < 1e-10 ? 1e-10 : sumExp;
      const probs = expLogits.map((v) => v / safeSum);

      // Combined gradient for all outputs (actor + critic)
      const combinedGrad = new Array(this.actionSize + 1).fill(0);

      // Actor gradient: -A * (one_hot(action) - probs)
      for (let j = 0; j < this.actionSize; j++) {
        combinedGrad[j] = -advantage * ((j === this.actions[i] ? 1 : 0) - probs[j]);
      }

      // Entropy bonus (maximize entropy)
      for (let j = 0; j < this.actionSize; j++) {
        if (probs[j] > 1e-10) {
          const entropyGrad = probs[j] * (this.computeEntropy(probs) + Math.log(probs[j]));
          combinedGrad[j] += this.entropyCoef * entropyGrad;
        }
      }

      // Critic gradient: MSE loss = 0.5 * (value - target)^2
      // dLoss/dvalue = value - target
      combinedGrad[this.actionSize] = value - normalizedReturns[i];

      // Clip gradients
      const clippedGrad = this.clipGradients(combinedGrad);

      const gradNorm = this.network.backwardOutputGradient(clippedGrad, this.learningRate);
      totalLoss += Math.abs(advantage) + Math.abs(gradNorm);
    }

    const avgLoss =
      this.states.length > 0 ? totalLoss / this.states.length : 0;

    // Clear memory
    this.states = [];
    this.actions = [];
    this.rewards = [];

    return avgLoss;
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

  getPolicy(): number[] {
    if (this.states.length === 0) {
      return new Array(this.actionSize).fill(1 / this.actionSize);
    }
    const output = this.network.forward(
      this.states[this.states.length - 1]
    );
    const logits = output.slice(0, this.actionSize);
    const maxLogit = Math.max(...logits);
    const expLogits = logits.map((v) =>
      Math.exp(Math.min(v - maxLogit, 50))
    );
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    return expLogits.map((v) => v / sumExp);
  }

  reset(): void {
    this.network = new NeuralNetwork([
      this.stateSize,
      64,
      64,
      this.actionSize + 1,
    ]);
    this.states = [];
    this.actions = [];
    this.rewards = [];
  }
}
