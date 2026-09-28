export class QLearning {
  qTable: Map<string, number[]>;
  learningRate: number;
  epsilon: number;
  epsilonDecay: number;
  discount: number;
  actionSize: number;
  minEpsilon = 0.01;

  constructor(
    _stateSize: number,
    actionSize: number,
    params: { learningRate: number; epsilon: number; discount: number; epsilonDecay?: number }
  ) {
    this.qTable = new Map();
    this.learningRate = params.learningRate;
    this.epsilon = params.epsilon;
    this.discount = params.discount;
    this.epsilonDecay = params.epsilonDecay ?? 0.995;
    this.actionSize = actionSize;
  }

  private stateKey(state: number[]): string {
    return state.map((v) => Math.round(v * 100) / 100).join(",");
  }

  private getQValues(state: number[]): number[] {
    const key = this.stateKey(state);
    if (!this.qTable.has(key)) {
      this.qTable.set(key, new Array(this.actionSize).fill(0));
    }
    return this.qTable.get(key)!;
  }

  selectAction(state: number[], epsilonOverride?: number): number {
    const eps = epsilonOverride ?? this.epsilon;
    if (Math.random() < eps) {
      return Math.floor(Math.random() * this.actionSize);
    }
    const qValues = this.getQValues(state);
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
    const qValues = this.getQValues(state);
    const nextQValues = this.getQValues(nextState);

    let maxNextQ = -Infinity;
    for (let i = 0; i < nextQValues.length; i++) {
      if (nextQValues[i] > maxNextQ) {
        maxNextQ = nextQValues[i];
      }
    }

    const target = reward + (done ? 0 : this.discount * maxNextQ);
    const tdError = target - qValues[action];
    qValues[action] += this.learningRate * tdError;

    // Decay epsilon
    this.epsilon = Math.max(this.minEpsilon, this.epsilon * this.epsilonDecay);

    return Math.abs(tdError);
  }

  getPolicy(): number[] {
    const avgQ = new Array(this.actionSize).fill(0);
    let count = 0;
    this.qTable.forEach((qValues) => {
      for (let i = 0; i < qValues.length; i++) {
        avgQ[i] += qValues[i];
      }
      count++;
    });
    if (count > 0) {
      for (let i = 0; i < avgQ.length; i++) {
        avgQ[i] /= count;
      }
    }
    // Softmax
    const maxVal = Math.max(...avgQ);
    const expQ = avgQ.map((v) => Math.exp(v - maxVal));
    const sumExp = expQ.reduce((a, b) => a + b, 0);
    return expQ.map((v) => v / sumExp);
  }

  reset(): void {
    this.qTable.clear();
    this.epsilon = 0.3;
  }
}
