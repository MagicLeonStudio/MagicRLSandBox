export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface CNNWeights {
  w1: Float32Array;
  b1: Float32Array;
  w2: Float32Array;
  b2: Float32Array;
  w3: Float32Array;
  b3: Float32Array;
  w4: Float32Array;
  b4: Float32Array;
}

export interface CNNActivations {
  conv1: Float32Array;
  conv2: Float32Array;
  dense: Float32Array;
}

const IMAGE = 12;
const CHANNELS = 3;
const KERNEL = 3;
const OUT1 = 5;
const FILTERS1 = 4;
const OUT2 = 2;
const FILTERS2 = 8;
const FLAT = OUT2 * OUT2 * FILTERS2;
const HIDDEN = 64;
const ACTIONS = 4;

/**
 * Tiny convolutional Q network, implemented with plain typed arrays.
 * 12x12x3 -> conv 4x3x3/s2 -> conv 8x3x3/s2 -> dense 64 -> 4 Q values.
 */
export class TinyCNN {
  w1 = new Float32Array(FILTERS1 * KERNEL * KERNEL * CHANNELS);
  b1 = new Float32Array(FILTERS1);
  w2 = new Float32Array(FILTERS2 * KERNEL * KERNEL * FILTERS1);
  b2 = new Float32Array(FILTERS2);
  w3 = new Float32Array(HIDDEN * FLAT);
  b3 = new Float32Array(HIDDEN);
  w4 = new Float32Array(ACTIONS * HIDDEN);
  b4 = new Float32Array(ACTIONS);

  private gw1 = new Float32Array(this.w1.length);
  private gb1 = new Float32Array(FILTERS1);
  private gw2 = new Float32Array(this.w2.length);
  private gb2 = new Float32Array(FILTERS2);
  private gw3 = new Float32Array(this.w3.length);
  private gb3 = new Float32Array(HIDDEN);
  private gw4 = new Float32Array(this.w4.length);
  private gb4 = new Float32Array(ACTIONS);

  private x = new Float32Array(IMAGE * IMAGE * CHANNELS);
  private z1 = new Float32Array(OUT1 * OUT1 * FILTERS1);
  private a1 = new Float32Array(this.z1.length);
  private z2 = new Float32Array(OUT2 * OUT2 * FILTERS2);
  private a2 = new Float32Array(this.z2.length);
  private flat = new Float32Array(FLAT);
  private z3 = new Float32Array(HIDDEN);
  private a3 = new Float32Array(HIDDEN);
  private out = new Float32Array(ACTIONS);

  private d4 = new Float32Array(ACTIONS);
  private d3 = new Float32Array(HIDDEN);
  private d2 = new Float32Array(this.z2.length);
  private da1 = new Float32Array(this.z1.length);
  private dz1 = new Float32Array(this.z1.length);

  constructor(seed = 1) {
    const rng = mulberry32(seed);
    const init = (arr: Float32Array, fanIn: number, scale = 1) => {
      const limit = Math.sqrt(2 / fanIn) * scale;
      for (let i = 0; i < arr.length; i++) arr[i] = (rng() * 2 - 1) * limit;
    };
    init(this.w1, KERNEL * KERNEL * CHANNELS);
    init(this.w2, KERNEL * KERNEL * FILTERS1);
    init(this.w3, FLAT);
    init(this.w4, HIDDEN, 0.25);
  }

  forward(input: Float32Array): number[] {
    this.x.set(input);

    for (let y = 0; y < OUT1; y++) {
      for (let x = 0; x < OUT1; x++) {
        for (let f = 0; f < FILTERS1; f++) {
          let sum = this.b1[f];
          for (let ky = 0; ky < KERNEL; ky++) {
            for (let kx = 0; kx < KERNEL; kx++) {
              for (let c = 0; c < CHANNELS; c++) {
                const xi = ((y * 2 + ky) * IMAGE + (x * 2 + kx)) * CHANNELS + c;
                const wi = ((f * KERNEL + ky) * KERNEL + kx) * CHANNELS + c;
                sum += this.w1[wi] * this.x[xi];
              }
            }
          }
          const oi = (y * OUT1 + x) * FILTERS1 + f;
          this.z1[oi] = sum;
          this.a1[oi] = Math.max(0, sum);
        }
      }
    }

    for (let y = 0; y < OUT2; y++) {
      for (let x = 0; x < OUT2; x++) {
        for (let f = 0; f < FILTERS2; f++) {
          let sum = this.b2[f];
          for (let ky = 0; ky < KERNEL; ky++) {
            for (let kx = 0; kx < KERNEL; kx++) {
              for (let c = 0; c < FILTERS1; c++) {
                const xi = ((y * 2 + ky) * OUT1 + (x * 2 + kx)) * FILTERS1 + c;
                const wi = ((f * KERNEL + ky) * KERNEL + kx) * FILTERS1 + c;
                sum += this.w2[wi] * this.a1[xi];
              }
            }
          }
          const oi = (y * OUT2 + x) * FILTERS2 + f;
          this.z2[oi] = sum;
          this.a2[oi] = Math.max(0, sum);
        }
      }
    }

    this.flat.set(this.a2);
    for (let h = 0; h < HIDDEN; h++) {
      let sum = this.b3[h];
      for (let i = 0; i < FLAT; i++) sum += this.w3[h * FLAT + i] * this.flat[i];
      this.z3[h] = sum;
      this.a3[h] = Math.max(0, sum);
    }
    for (let a = 0; a < ACTIONS; a++) {
      let sum = this.b4[a];
      for (let h = 0; h < HIDDEN; h++) sum += this.w4[a * HIDDEN + h] * this.a3[h];
      this.out[a] = sum;
    }
    return Array.from(this.out);
  }

  getActivations(): CNNActivations {
    return { conv1: this.a1, conv2: this.a2, dense: this.a3 };
  }

  getWeights(): CNNWeights {
    return {
      w1: this.w1.slice(), b1: this.b1.slice(),
      w2: this.w2.slice(), b2: this.b2.slice(),
      w3: this.w3.slice(), b3: this.b3.slice(),
      w4: this.w4.slice(), b4: this.b4.slice(),
    };
  }

  setWeights(w: CNNWeights) {
    this.w1.set(w.w1); this.b1.set(w.b1);
    this.w2.set(w.w2); this.b2.set(w.b2);
    this.w3.set(w.w3); this.b3.set(w.b3);
    this.w4.set(w.w4); this.b4.set(w.b4);
  }

  /** One-sample semi-gradient update. The caller supplies the effective LR. */
  backward(target: number[], learningRate: number): number {
    this.gw1.fill(0); this.gb1.fill(0);
    this.gw2.fill(0); this.gb2.fill(0);
    this.gw3.fill(0); this.gb3.fill(0);
    this.gw4.fill(0); this.gb4.fill(0);
    this.da1.fill(0);

    let loss = 0;
    for (let a = 0; a < ACTIONS; a++) {
      const d = this.out[a] - target[a];
      this.d4[a] = d;
      loss += 0.5 * d * d;
      this.gb4[a] = d;
      for (let h = 0; h < HIDDEN; h++) this.gw4[a * HIDDEN + h] = d * this.a3[h];
    }

    for (let h = 0; h < HIDDEN; h++) {
      let d = 0;
      for (let a = 0; a < ACTIONS; a++) d += this.d4[a] * this.w4[a * HIDDEN + h];
      d = this.z3[h] > 0 ? d : 0;
      this.d3[h] = d;
      this.gb3[h] = d;
      for (let i = 0; i < FLAT; i++) this.gw3[h * FLAT + i] = d * this.flat[i];
    }

    for (let i = 0; i < FLAT; i++) {
      let d = 0;
      for (let h = 0; h < HIDDEN; h++) d += this.d3[h] * this.w3[h * FLAT + i];
      this.d2[i] = this.z2[i] > 0 ? d : 0;
    }

    for (let y = 0; y < OUT2; y++) {
      for (let x = 0; x < OUT2; x++) {
        for (let f = 0; f < FILTERS2; f++) {
          const oi = (y * OUT2 + x) * FILTERS2 + f;
          const g = this.d2[oi];
          this.gb2[f] += g;
          for (let ky = 0; ky < KERNEL; ky++) {
            for (let kx = 0; kx < KERNEL; kx++) {
              for (let c = 0; c < FILTERS1; c++) {
                const xi = ((y * 2 + ky) * OUT1 + (x * 2 + kx)) * FILTERS1 + c;
                const wi = ((f * KERNEL + ky) * KERNEL + kx) * FILTERS1 + c;
                this.gw2[wi] += g * this.a1[xi];
                this.da1[xi] += g * this.w2[wi];
              }
            }
          }
        }
      }
    }

    for (let i = 0; i < this.dz1.length; i++) {
      this.dz1[i] = this.z1[i] > 0 ? this.da1[i] : 0;
    }

    for (let y = 0; y < OUT1; y++) {
      for (let x = 0; x < OUT1; x++) {
        for (let f = 0; f < FILTERS1; f++) {
          const oi = (y * OUT1 + x) * FILTERS1 + f;
          const g = this.dz1[oi];
          this.gb1[f] += g;
          for (let ky = 0; ky < KERNEL; ky++) {
            for (let kx = 0; kx < KERNEL; kx++) {
              for (let c = 0; c < CHANNELS; c++) {
                const xi = ((y * 2 + ky) * IMAGE + (x * 2 + kx)) * CHANNELS + c;
                const wi = ((f * KERNEL + ky) * KERNEL + kx) * CHANNELS + c;
                this.gw1[wi] += g * this.x[xi];
              }
            }
          }
        }
      }
    }

    let norm = 0;
    const grads = [this.gw1, this.gb1, this.gw2, this.gb2, this.gw3, this.gb3, this.gw4, this.gb4];
    for (const arr of grads) for (let i = 0; i < arr.length; i++) norm += arr[i] * arr[i];
    norm = Math.sqrt(norm);
    const scale = norm > 10 ? 10 / norm : 1;
    const lr = learningRate * scale;

    for (let i = 0; i < this.w1.length; i++) this.w1[i] -= lr * this.gw1[i];
    for (let i = 0; i < this.b1.length; i++) this.b1[i] -= lr * this.gb1[i];
    for (let i = 0; i < this.w2.length; i++) this.w2[i] -= lr * this.gw2[i];
    for (let i = 0; i < this.b2.length; i++) this.b2[i] -= lr * this.gb2[i];
    for (let i = 0; i < this.w3.length; i++) this.w3[i] -= lr * this.gw3[i];
    for (let i = 0; i < this.b3.length; i++) this.b3[i] -= lr * this.gb3[i];
    for (let i = 0; i < this.w4.length; i++) this.w4[i] -= lr * this.gw4[i];
    for (let i = 0; i < this.b4.length; i++) this.b4[i] -= lr * this.gb4[i];
    return loss;
  }
}
