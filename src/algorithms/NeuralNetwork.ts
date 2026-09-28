export class NeuralNetwork {
  weights: number[][][];
  biases: number[][];
  layerSizes: number[];
  activations: number[][];
  zs: number[][];

  constructor(layerSizes: number[]) {
    this.layerSizes = layerSizes;
    this.weights = [];
    this.biases = [];
    this.activations = [];
    this.zs = [];

    for (let i = 1; i < layerSizes.length; i++) {
      const w: number[][] = [];
      for (let j = 0; j < layerSizes[i]; j++) {
        const row: number[] = [];
        for (let k = 0; k < layerSizes[i - 1]; k++) {
          // He initialization for ReLU
          row.push((Math.random() * 2 - 1) * Math.sqrt(2.0 / layerSizes[i - 1]));
        }
        w.push(row);
      }
      this.weights.push(w);
      this.biases.push(new Array(layerSizes[i]).fill(0));
    }
  }

  forward(input: number[]): number[] {
    this.activations = [input];
    this.zs = [];
    let a = input;

    for (let i = 0; i < this.weights.length; i++) {
      const z: number[] = [];
      for (let j = 0; j < this.biases[i].length; j++) {
        let sum = this.biases[i][j];
        for (let k = 0; k < a.length; k++) {
          sum += this.weights[i][j][k] * a[k];
        }
        z.push(sum);
      }
      this.zs.push(z);

      if (i === this.weights.length - 1) {
        // Output layer: linear activation
        a = z;
      } else {
        // Hidden layers: ReLU
        a = z.map((v) => Math.max(0, v));
      }
      this.activations.push(a);
    }

    return a;
  }

  /**
   * Backpropagation for MSE regression target.
   * Used by: DQN (critic), PPO (critic)
   * target[i] = desired output for unit i
   */
  backward(target: number[], learningRate: number): number {
    const output = this.activations[this.activations.length - 1];
    let loss = 0;
    const deltas: number[][] = [];

    // Output layer delta (MSE loss gradient = output - target)
    const outputDelta: number[] = [];
    for (let i = 0; i < output.length; i++) {
      const err = output[i] - target[i];
      loss += 0.5 * err * err;
      outputDelta.push(err);
    }
    deltas.push(outputDelta);

    // Backpropagate through hidden layers
    for (let i = this.weights.length - 2; i >= 0; i--) {
      const delta: number[] = [];
      for (let j = 0; j < this.activations[i + 1].length; j++) {
        let err = 0;
        for (let k = 0; k < deltas[deltas.length - 1].length; k++) {
          err += deltas[deltas.length - 1][k] * this.weights[i + 1][k][j];
        }
        // ReLU derivative
        delta.push(err * (this.zs[i][j] > 0 ? 1 : 0));
      }
      deltas.push(delta);
    }

    deltas.reverse();

    // Update weights and biases (gradient descent)
    for (let i = 0; i < this.weights.length; i++) {
      for (let j = 0; j < this.weights[i].length; j++) {
        for (let k = 0; k < this.weights[i][j].length; k++) {
          this.weights[i][j][k] -= learningRate * deltas[i][j] * this.activations[i][k];
        }
        this.biases[i][j] -= learningRate * deltas[i][j];
      }
    }

    return loss;
  }

  /**
   * Backpropagation with directly specified output layer gradients.
   * Used by: REINFORCE, A3C, ActorCritic, PPO (actor)
   *
   * outputGrad[i] = d(Loss) / d(output_i)
   * For policy gradient: outputGrad = -advantage * (one_hot(action) - probs)
   */
  backwardOutputGradient(outputGrad: number[], learningRate: number): number {
    const deltas: number[][] = [];

    // Output layer: linear activation, derivative = 1
    // delta_output = outputGrad * 1 = outputGrad
    deltas.push([...outputGrad]);

    // Backpropagate through hidden layers
    for (let i = this.weights.length - 2; i >= 0; i--) {
      const delta: number[] = [];
      for (let j = 0; j < this.activations[i + 1].length; j++) {
        let err = 0;
        for (let k = 0; k < deltas[deltas.length - 1].length; k++) {
          err += deltas[deltas.length - 1][k] * this.weights[i + 1][k][j];
        }
        // ReLU derivative
        delta.push(err * (this.zs[i][j] > 0 ? 1 : 0));
      }
      deltas.push(delta);
    }

    deltas.reverse();

    // Update weights and biases (gradient descent)
    let gradNorm = 0;
    for (let i = 0; i < this.weights.length; i++) {
      for (let j = 0; j < this.weights[i].length; j++) {
        for (let k = 0; k < this.weights[i][j].length; k++) {
          const grad = deltas[i][j] * this.activations[i][k];
          gradNorm += grad * grad;
          this.weights[i][j][k] -= learningRate * grad;
        }
        this.biases[i][j] -= learningRate * deltas[i][j];
      }
    }

    return Math.sqrt(gradNorm);
  }

  getWeights(): { weights: number[][][]; biases: number[][] } {
    return {
      weights: this.weights.map((w) => w.map((row) => [...row])),
      biases: this.biases.map((b) => [...b]),
    };
  }

  setWeights(weights: number[][][], biases: number[][]) {
    this.weights = weights.map((w) => w.map((row) => [...row]));
    this.biases = biases.map((b) => [...b]);
  }
}
