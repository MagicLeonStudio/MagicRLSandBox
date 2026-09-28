import { NeuralNetwork } from "./NeuralNetwork";

interface Experience {
  state: number[];
  action: number;
  reward: number;
  nextState: number[];
  done: boolean;
}

export class DQN {
  network: NeuralNetwork;
  targetNetwork: NeuralNetwork;
  replayBuffer: Experience[];
  learningRate: number;
  epsilon: number;
  epsilonDecay: number;
  batchSize: number;
  gamma: number;
  replayBufferSize: number;
  targetUpdateFreq: number;
  actionSize: number;
  stateSize: number;
  stepCount = 0;
  minEpsilon = 0.01;

  constructor(
    stateSize: number,
    actionSize: number,
    params: {
      learningRate: number;
      epsilon: number;
      batchSize: number;
      replayBuffer?: number;
      targetUpdate?: number;
      epsilonDecay?: number;
    }
  ) {
    this.stateSize = stateSize;
    this.actionSize = actionSize;
    this.network = new NeuralNetwork([stateSize, 64, 64, actionSize]);
    this.targetNetwork = new NeuralNetwork([stateSize, 64, 64, actionSize]);
    this.replayBuffer = [];
    this.learningRate = params.learningRate;
    this.epsilon = params.epsilon;
    this.epsilonDecay = params.epsilonDecay ?? 0.995;
    this.batchSize = params.batchSize;
    this.gamma = 0.99;
    this.replayBufferSize = params.replayBuffer ?? 5000;
    this.targetUpdateFreq = params.targetUpdate ?? 100;
  }

  selectAction(state: number[], epsilonOverride?: number): number {
    const eps = epsilonOverride ?? this.epsilon;
    if (Math.random() < eps) {
      return Math.floor(Math.random() * this.actionSize);
    }
    const qValues = this.network.forward(state);
    let maxVal = -Infinity;
    let bestAction = 0;
    for (let i = 0; i < qValues.length; i++) {
      if (qValues[i] > maxVal) {
        maxVal = qValues[i];
        bestAction = i;
      }
    }
    return bestAction;
  }

  train(
    state: number[],
    action: number,
    reward: number,
    nextState: number[],
    done: boolean
  ): number {
    // Store experience
    this.replayBuffer.push({
      state: [...state],
      action,
      reward,
      nextState: [...nextState],
      done,
    });
    if (this.replayBuffer.length > this.replayBufferSize) {
      this.replayBuffer.shift();
    }

    this.stepCount++;

    // Decay epsilon (per step, mild)
    this.epsilon = Math.max(
      this.minEpsilon,
      this.epsilon * this.epsilonDecay
    );

    // Update target network periodically
    if (this.stepCount % this.targetUpdateFreq === 0) {
      const w = this.network.getWeights();
      this.targetNetwork.setWeights(w.weights, w.biases);
    }

    // Train only if we have enough samples
    if (this.replayBuffer.length < this.batchSize) {
      return 0;
    }

    // Sample mini-batch
    const batch: Experience[] = [];
    for (let i = 0; i < this.batchSize; i++) {
      const idx = Math.floor(Math.random() * this.replayBuffer.length);
      batch.push(this.replayBuffer[idx]);
    }

    // Accumulate gradients and apply averaged update
    let totalLoss = 0;
    const lrPerSample = this.learningRate / this.batchSize;

    // Store weight deltas for batch update
    const weightDeltas = this.network.weights.map((layer) =>
      layer.map((row) => new Array(row.length).fill(0))
    );
    const biasDeltas = this.network.biases.map((b) => new Array(b.length).fill(0));

    for (const exp of batch) {
      const qValues = this.network.forward(exp.state);
      const nextQValues = this.targetNetwork.forward(exp.nextState);

      let maxNextQ = -Infinity;
      for (let i = 0; i < nextQValues.length; i++) {
        if (nextQValues[i] > maxNextQ) {
          maxNextQ = nextQValues[i];
        }
      }

      const target =
        exp.reward + (exp.done ? 0 : this.gamma * maxNextQ);
      const targetQ = [...qValues];
      targetQ[exp.action] = target;

      // Compute loss
      const tdError = target - qValues[exp.action];
      totalLoss += 0.5 * tdError * tdError;

      // Manual backprop for this sample (accumulate)
      this._accumulateGradient(targetQ, weightDeltas, biasDeltas);
    }

    // Apply averaged gradients
    for (let i = 0; i < this.network.weights.length; i++) {
      for (let j = 0; j < this.network.weights[i].length; j++) {
        for (let k = 0; k < this.network.weights[i][j].length; k++) {
          this.network.weights[i][j][k] -=
            lrPerSample * weightDeltas[i][j][k];
        }
        this.network.biases[i][j] -=
          lrPerSample * biasDeltas[i][j];
      }
    }

    return totalLoss / this.batchSize;
  }

  /**
   * Accumulate gradient for one sample into weightDeltas and biasDeltas.
   * Does NOT update weights.
   */
  private _accumulateGradient(
    target: number[],
    weightDeltas: number[][][],
    biasDeltas: number[][]
  ): void {
    const output = this.network.activations[this.network.activations.length - 1];
    const deltas: number[][] = [];

    // Output layer delta
    const outputDelta: number[] = [];
    for (let i = 0; i < output.length; i++) {
      outputDelta.push(output[i] - target[i]);
    }
    deltas.push(outputDelta);

    // Backpropagate through hidden layers
    for (let i = this.network.weights.length - 2; i >= 0; i--) {
      const delta: number[] = [];
      for (let j = 0; j < this.network.activations[i + 1].length; j++) {
        let err = 0;
        for (let k = 0; k < deltas[deltas.length - 1].length; k++) {
          err +=
            deltas[deltas.length - 1][k] *
            this.network.weights[i + 1][k][j];
        }
        delta.push(err * (this.network.zs[i][j] > 0 ? 1 : 0));
      }
      deltas.push(delta);
    }

    deltas.reverse();

    // Accumulate into deltas arrays
    for (let i = 0; i < this.network.weights.length; i++) {
      for (let j = 0; j < this.network.weights[i].length; j++) {
        for (let k = 0; k < this.network.weights[i][j].length; k++) {
          weightDeltas[i][j][k] +=
            deltas[i][j] * this.network.activations[i][k];
        }
        biasDeltas[i][j] += deltas[i][j];
      }
    }
  }

  getPolicy(): number[] {
    if (this.replayBuffer.length === 0) {
      return new Array(this.actionSize).fill(1 / this.actionSize);
    }
    const sample =
      this.replayBuffer[
        Math.floor(Math.random() * this.replayBuffer.length)
      ];
    const qValues = this.network.forward(sample.state);
    const maxVal = Math.max(...qValues);
    const expQ = qValues.map((v) => Math.exp(Math.min(v - maxVal, 50)));
    const sumExp = expQ.reduce((a, b) => a + b, 0);
    return expQ.map((v) => v / sumExp);
  }

  reset(): void {
    this.network = new NeuralNetwork([this.stateSize, 64, 64, this.actionSize]);
    this.targetNetwork = new NeuralNetwork([
      this.stateSize,
      64,
      64,
      this.actionSize,
    ]);
    this.replayBuffer = [];
    this.stepCount = 0;
    this.epsilon = 1.0;
  }
}
