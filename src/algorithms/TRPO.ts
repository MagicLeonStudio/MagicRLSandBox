import { NeuralNetwork } from "./NeuralNetwork";

export class TRPO {
  actor: NeuralNetwork;
  critic: NeuralNetwork;
  learningRate: number;
  gamma: number;
  delta: number;
  actionSize: number;
  stateSize: number;

  // Episode memory
  states: number[][];
  actions: number[];
  rewards: number[];

  constructor(
    stateSize: number,
    actionSize: number,
    params: { learningRate: number; gamma: number; delta?: number }
  ) {
    this.stateSize = stateSize;
    this.actionSize = actionSize;
    this.actor = new NeuralNetwork([stateSize, 64, 64, actionSize]);
    this.critic = new NeuralNetwork([stateSize, 64, 64, 1]);
    this.learningRate = params.learningRate;
    this.gamma = params.gamma;
    this.delta = params.delta ?? 0.01;
    this.states = [];
    this.actions = [];
    this.rewards = [];
  }

  selectAction(state: number[], _epsilonOverride?: number): number {
    const logits = this.actor.forward(state);
    const maxLogit = Math.max(...logits);
    const expLogits = logits.map((v) => Math.exp(Math.min(v - maxLogit, 50)));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    const probs = expLogits.map((v) => v / sumExp);

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
      // Critic update
      const value = this.critic.forward(this.states[i])[0];
      const criticTarget = [normalizedReturns[i]];
      const criticLoss = this.critic.backward(
        criticTarget,
        this.learningRate
      );

      const advantage = normalizedReturns[i] - value;

      // Actor: natural policy gradient (simplified TRPO)
      const logits = this.actor.forward(this.states[i]);
      const maxLogit = Math.max(...logits);
      const expLogits = logits.map((v) =>
        Math.exp(Math.min(v - maxLogit, 50))
      );
      const sumExp = expLogits.reduce((a, b) => a + b, 0);
      const probs = expLogits.map((v) => v / sumExp);

      // Policy gradient with trust region (simplified):
      // Instead of full conjugate gradient, we use a small fixed step size
      // which approximates the trust region constraint
      const actorGrad = new Array(this.actionSize).fill(0);
      for (let j = 0; j < this.actionSize; j++) {
        actorGrad[j] =
          -advantage * ((j === this.actions[i] ? 1 : 0) - probs[j]);
      }

      const gradNorm = this.actor.backwardOutputGradient(
        actorGrad,
        this.learningRate
      );
      totalLoss += criticLoss + Math.abs(gradNorm);
    }

    const avgLoss =
      this.states.length > 0 ? totalLoss / this.states.length : 0;

    this.states = [];
    this.actions = [];
    this.rewards = [];

    return avgLoss;
  }

  getPolicy(): number[] {
    if (this.states.length === 0) {
      return new Array(this.actionSize).fill(1 / this.actionSize);
    }
    const logits = this.actor.forward(this.states[this.states.length - 1]);
    const maxLogit = Math.max(...logits);
    const expLogits = logits.map((v) =>
      Math.exp(Math.min(v - maxLogit, 50))
    );
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    return expLogits.map((v) => v / sumExp);
  }

  reset(): void {
    this.actor = new NeuralNetwork([this.stateSize, 64, 64, this.actionSize]);
    this.critic = new NeuralNetwork([this.stateSize, 64, 64, 1]);
    this.states = [];
    this.actions = [];
    this.rewards = [];
  }
}
