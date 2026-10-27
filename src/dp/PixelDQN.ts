import { TinyCNN, mulberry32 } from "@/dp/TinyCNN";
import type { CNNActivations } from "@/dp/TinyCNN";

export interface PixelExperience {
  id: number;
  episode: number;
  step: number;
  s: Float32Array;
  a: number;
  r: number;
  s2: Float32Array;
  done: boolean;
  nextValid: number[];
}

export interface TrainInfo {
  target: number;
  prediction: number;
  tdError: number;
  maxNext: number;
  reward: number;
  loss: number;
}

export interface ProbePoint {
  step: number;
  online: number;
  reference: number;
}

export interface PixelDQNParams {
  seed: number;
  useReplay: boolean;
  useTargetNet: boolean;
  lr?: number;
  gamma?: number;
  batchSize?: number;
  bufferSize?: number;
  targetUpdate?: number;
}

const GRID = 6;
const CELL = 2;
const IMAGE = GRID * CELL;
const CHANNELS = 3;
const MAX_STEPS = 48;
const WALLS = new Set<number>();

function hash01(n: number): number {
  let x = n | 0;
  x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
  x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
  return ((x ^ (x >>> 16)) >>> 0) / 4294967296;
}

/** Grid dynamics with a real pixel observation, not privileged coordinates. */
export class PixelGridEnv {
  pos = 0;
  goal = 30;
  steps = 0;
  episode = 0;
  private rng: () => number;
  shaping = 0.16;
  layout: "fixed" | "randomStart" | "randomBoth" = "fixed";

  constructor(seed = 1) {
    this.rng = mulberry32(seed);
    this.reset();
  }

  reset(): Float32Array {
    this.episode++;
    this.steps = 0;
    const free: number[] = [];
    for (let i = 0; i < GRID * GRID; i++) if (!WALLS.has(i)) free.push(i);
    const distance = () => {
      const pr = Math.floor(this.pos / GRID), pc = this.pos % GRID;
      const gr = Math.floor(this.goal / GRID), gc = this.goal % GRID;
      return Math.abs(pr - gr) + Math.abs(pc - gc);
    };
    if (this.layout === "fixed") {
      this.pos = 0;
      this.goal = 35;
    } else {
      this.goal = this.layout === "randomStart" ? 35 : free[Math.floor(this.rng() * free.length)];
      this.pos = free[Math.floor(this.rng() * free.length)];
      let guard = 0;
      while ((this.goal === this.pos || distance() < 3) && guard++ < 100) {
        this.pos = free[Math.floor(this.rng() * free.length)];
        if (this.layout === "randomBoth") this.goal = free[Math.floor(this.rng() * free.length)];
      }
    }
    return this.obs();
  }

  obs(): Float32Array {
    const out = new Float32Array(IMAGE * IMAGE * CHANNELS);
    const textureSeed = this.episode * 7919 + 17;
    for (let i = 0; i < IMAGE * IMAGE; i++) {
      const n = hash01(textureSeed + i * 13);
      out[i * 3] = 0.012 + n * 0.025;
      out[i * 3 + 1] = 0.009 + n * 0.014;
      out[i * 3 + 2] = 0.028 + n * 0.045;
    }
    const paint = (cell: number, rgb: [number, number, number]) => {
      const row = Math.floor(cell / GRID), col = cell % GRID;
      for (let dy = 0; dy < CELL; dy++) {
        for (let dx = 0; dx < CELL; dx++) {
          const p = ((row * CELL + dy) * IMAGE + col * CELL + dx) * CHANNELS;
          out[p] = rgb[0]; out[p + 1] = rgb[1]; out[p + 2] = rgb[2];
        }
      }
    };
    for (const wall of WALLS) paint(wall, [0.17, 0.14, 0.28]);
    paint(this.goal, [0.55, 0.36, 0.96]);
    paint(this.pos, [1.0, 0.72, 0.04]);
    return out;
  }

  validActions(): number[] {
    const row = Math.floor(this.pos / GRID), col = this.pos % GRID;
    const valid: number[] = [];
    const candidates: Array<[number, number, number]> = [
      [0, row - 1, col], [1, row + 1, col], [2, row, col - 1], [3, row, col + 1],
    ];
    for (const [action, nr, nc] of candidates) {
      if (nr < 0 || nr >= GRID || nc < 0 || nc >= GRID) continue;
      if (!WALLS.has(nr * GRID + nc)) valid.push(action);
    }
    return valid;
  }

  step(action: number): { s2: Float32Array; r: number; done: boolean } {
    const row = Math.floor(this.pos / GRID), col = this.pos % GRID;
    const goalRow = Math.floor(this.goal / GRID), goalCol = this.goal % GRID;
    const oldDistance = Math.abs(row - goalRow) + Math.abs(col - goalCol);
    let nr = row, nc = col;
    if (action === 0) nr = Math.max(0, row - 1);
    else if (action === 1) nr = Math.min(GRID - 1, row + 1);
    else if (action === 2) nc = Math.max(0, col - 1);
    else nc = Math.min(GRID - 1, col + 1);
    const next = nr * GRID + nc;
    this.steps++;
    if (WALLS.has(next) || next === this.pos) {
      return { s2: this.obs(), r: -0.02, done: this.steps >= MAX_STEPS };
    }
    this.pos = next;
    const newDistance = Math.abs(nr - goalRow) + Math.abs(nc - goalCol);
    const shaping = this.shaping * (oldDistance - newDistance);
    if (next === this.goal) return { s2: this.obs(), r: 1 + shaping, done: true };
    return { s2: this.obs(), r: -0.02 + shaping, done: this.steps >= MAX_STEPS };
  }
}

/** A compact DQN with independently switchable replay and target network. */
export class PixelDQN {
  online: TinyCNN;
  target: TinyCNN;
  params: Required<Omit<PixelDQNParams, "seed">> & { seed: number };
  buffer: PixelExperience[] = [];
  lastBatch: PixelExperience[] = [];
  lastTrain: TrainInfo | null = null;
  probeHistory: ProbePoint[] = [];
  stepCount = 0;
  lastSyncStep = 0;
  epsilon = 1;
  private nextId = 1;
  private rng: () => number;
  private probe: Float32Array | null = null;

  constructor(params: PixelDQNParams) {
    this.params = {
      seed: params.seed,
      useReplay: params.useReplay,
      useTargetNet: params.useTargetNet,
      lr: params.lr ?? 0.003,
      gamma: params.gamma ?? 0.95,
      batchSize: params.batchSize ?? 16,
      bufferSize: params.bufferSize ?? 3000,
      targetUpdate: params.targetUpdate ?? 100,
    };
    this.online = new TinyCNN(params.seed);
    this.target = new TinyCNN(params.seed);
    this.target.setWeights(this.online.getWeights());
    this.rng = mulberry32(params.seed + 100003);
  }

  selectAction(s: Float32Array, epsilon: number, valid = [0, 1, 2, 3]): number {
    if (this.rng() < epsilon) return valid[Math.floor(this.rng() * valid.length)];
    const q = this.online.forward(s);
    let best = valid[0];
    for (const action of valid) if (q[action] > q[best]) best = action;
    return best;
  }

  finishEpisode() {
    this.epsilon = Math.max(0.05, this.epsilon * 0.99);
  }

  setProbe(s: Float32Array) {
    this.probe = s.slice();
    this.probeHistory = [];
  }

  inspect(s: Float32Array): { q: number[]; activations: CNNActivations } {
    const q = this.online.forward(s);
    return { q, activations: this.online.getActivations() };
  }

  referenceQ(s: Float32Array): number[] {
    return (this.params.useTargetNet ? this.target : this.online).forward(s);
  }

  syncCountdown(): number {
    if (!this.params.useTargetNet) return 0;
    return this.params.targetUpdate - (this.stepCount % this.params.targetUpdate);
  }

  trainStep(
    s: Float32Array,
    a: number,
    r: number,
    s2: Float32Array,
    done: boolean,
    episode: number,
    step: number,
    nextValid: number[],
  ): number {
    this.stepCount++;
    const exp: PixelExperience = {
      id: this.nextId++, episode, step,
      s: s.slice(), a, r, s2: s2.slice(), done, nextValid: [...nextValid],
    };
    this.buffer.push(exp);
    if (this.buffer.length > this.params.bufferSize) this.buffer.shift();

    if (this.params.useTargetNet && this.stepCount % this.params.targetUpdate === 0) {
      this.target.setWeights(this.online.getWeights());
      this.lastSyncStep = this.stepCount;
    }

    let loss = 0;
    if (!this.params.useReplay) {
      this.lastBatch = [exp];
      loss = this.update(exp, this.params.lr);
    } else if (this.buffer.length >= this.params.batchSize) {
      const batch: PixelExperience[] = [];
      for (let i = 0; i < this.params.batchSize; i++) {
        batch.push(this.buffer[Math.floor(this.rng() * this.buffer.length)]);
      }
      this.lastBatch = batch;
      for (const item of batch) loss += this.update(item, this.params.lr / this.params.batchSize);
      loss /= this.params.batchSize;
    } else {
      this.lastBatch = [];
    }
    if (this.stepCount % 4 === 0) this.recordProbe();
    return loss;
  }

  private update(exp: PixelExperience, lr: number): number {
    const nextQ = (this.params.useTargetNet ? this.target : this.online).forward(exp.s2);
    const maxNext = exp.done ? 0 : Math.max(...exp.nextValid.map((action) => nextQ[action]));
    const q = this.online.forward(exp.s);
    const targetQ = [...q];
    const y = exp.r + this.params.gamma * maxNext;
    targetQ[exp.a] = y;
    const prediction = q[exp.a];
    const loss = this.online.backward(targetQ, lr);
    this.lastTrain = {
      target: y,
      prediction,
      tdError: y - prediction,
      maxNext,
      reward: exp.r,
      loss,
    };
    return loss;
  }

  private recordProbe() {
    if (!this.probe) return;
    const online = this.online.forward(this.probe)[3];
    const reference = (this.params.useTargetNet ? this.target : this.online).forward(this.probe)[3];
    this.probeHistory.push({ step: this.stepCount, online, reference });
    if (this.probeHistory.length > 160) this.probeHistory.shift();
  }
}

export const PIXEL_IMAGE_SIZE = IMAGE;
export const PIXEL_GRID_SIZE = GRID;
export const PIXEL_MAX_STEPS = MAX_STEPS;
