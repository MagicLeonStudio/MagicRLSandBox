import { NeuralNetwork } from "./NeuralNetwork";

export class REINFORCE {
  network: NeuralNetwork;
  learningRate: number;
  gamma: number;
  actionSize: number;
  stateSize: number;

  // Episode memory
  states: number[][];
  actions: number[];
  rewards: number[];

  constructor(
    stateSize: number,
    actionSize: number,
    params: { learningRate: number; gamma: number }
  ) {
    this.stateSize = stateSize;
    this.actionSize = actionSize;
    this.network = new NeuralNetwork([stateSize, 64, 64, actionSize]);
    this.learningRate = params.learningRate;
    this.gamma = params.gamma;
    this.states = [];
    this.actions = [];
    this.rewards = [];
  }

  selectAction(state: number[], _epsilonOverride?: number): number {
    const logits = this.network.forward(state);
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

    // Compute discounted returns
    const returns: number[] = [];
    let G = 0;
    for (let i = this.rewards.length - 1; i >= 0; i--) {
      G = this.rewards[i] + this.gamma * G;
      returns.unshift(G);
    }

    // Normalize returns for stability
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
      const logits = this.network.forward(this.states[i]);

      // Compute policy probabilities
      const maxLogit = Math.max(...logits);
      const expLogits = logits.map((v) => Math.exp(Math.min(v - maxLogit, 50)));
      const sumExp = expLogits.reduce((a, b) => a + b, 0);
      const probs = expLogits.map((v) => v / sumExp);

      // Policy gradient (for softmax policy):
      // ∂(-G * log π(a)) / ∂z_j = -G * (1{j==a} - π(j))
      const outputGrad = new Array(this.actionSize).fill(0);
      for (let j = 0; j < this.actionSize; j++) {
        outputGrad[j] =
          -normalizedReturns[i] * ((j === this.actions[i] ? 1 : 0) - probs[j]);
      }

      const gradNorm = this.network.backwardOutputGradient(
        outputGrad,
        this.learningRate
      );
      totalLoss += gradNorm;
    }

    const avgLoss =
      this.states.length > 0 ? totalLoss / this.states.length : 0;

    // Clear memory
    this.states = [];
    this.actions = [];
    this.rewards = [];

    return avgLoss;
  }

  getPolicy(): number[] {
    if (this.states.length === 0) {
      return new Array(this.actionSize).fill(1 / this.actionSize);
    }
    const logits = this.network.forward(
      this.states[this.states.length - 1]
    );
    const maxLogit = Math.max(...logits);
    const expLogits = logits.map((v) =>
      Math.exp(Math.min(v - maxLogit, 50))
    );
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    return expLogits.map((v) => v / sumExp);
  }

  reset(): void {
    this.network = new NeuralNetwork([this.stateSize, 64, 64, this.actionSize]);
    this.states = [];
    this.actions = [];
    this.rewards = [];
  }
}
