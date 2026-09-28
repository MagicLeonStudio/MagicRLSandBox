export interface CartPoleState {
  cartPosition: number;
  cartVelocity: number;
  poleAngle: number;
  poleVelocity: number;
  steps: number;
}

export type CartPoleAction = 0 | 1; // Push Left, Push Right

export const CARTPOLE_ACTIONS = ["Push Left", "Push Right"] as const;

export class CartPoleEngine {
  gravity = 9.8;
  cartMass = 1.0;
  poleMass = 0.1;
  totalMass = 1.1;
  poleLength = 0.5; // half-length
  forceMag = 10.0;
  tau = 0.02; // time step

  // Thresholds
  thetaThreshold = 0.21; // ~12 degrees
  xThreshold = 2.4;

  state: CartPoleState;

  constructor() {
    this.state = this.reset();
  }

  reset(): CartPoleState {
    this.state = {
      cartPosition: (Math.random() - 0.5) * 0.1,
      cartVelocity: (Math.random() - 0.5) * 0.1,
      poleAngle: (Math.random() - 0.5) * 0.1,
      poleVelocity: (Math.random() - 0.5) * 0.1,
      steps: 0,
    };
    return this.state;
  }

  step(action: CartPoleAction): {
    state: number[];
    reward: number;
    done: boolean;
    rawState: CartPoleState;
  } {
    let { cartPosition, cartVelocity, poleAngle, poleVelocity } = this.state;

    const force = action === 1 ? this.forceMag : -this.forceMag;
    const cosTheta = Math.cos(poleAngle);
    const sinTheta = Math.sin(poleAngle);

    const temp =
      (force + this.poleMass * this.poleLength * poleVelocity * poleVelocity * sinTheta) /
      this.totalMass;
    const thetaAcc =
      (this.gravity * sinTheta - cosTheta * temp) /
      (this.poleLength *
        (4.0 / 3.0 - (this.poleMass * cosTheta * cosTheta) / this.totalMass));
    const xAcc = temp - (this.poleMass * this.poleLength * thetaAcc * cosTheta) / this.totalMass;

    // Euler integration
    cartPosition += this.tau * cartVelocity;
    cartVelocity += this.tau * xAcc;
    poleAngle += this.tau * poleVelocity;
    poleVelocity += this.tau * thetaAcc;

    this.state.cartPosition = cartPosition;
    this.state.cartVelocity = cartVelocity;
    this.state.poleAngle = poleAngle;
    this.state.poleVelocity = poleVelocity;
    this.state.steps++;

    const done =
      Math.abs(cartPosition) > this.xThreshold ||
      Math.abs(poleAngle) > this.thetaThreshold;

    return {
      state: this.getObservation(),
      reward: 1.0,
      done,
      rawState: { ...this.state },
    };
  }

  getObservation(): number[] {
    return [
      this.state.cartPosition,
      this.state.cartVelocity,
      this.state.poleAngle,
      this.state.poleVelocity,
    ];
  }

  getStateSize(): number {
    return 4;
  }

  getActionSize(): number {
    return 2;
  }

  render(canvas: HTMLCanvasElement): void {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;

    // Background
    ctx.fillStyle = "#0D0D15";
    ctx.fillRect(0, 0, w, h);

    // Subtle horizontal grid lines
    ctx.strokeStyle = "rgba(30, 30, 46, 0.3)";
    ctx.lineWidth = 1;
    for (let y = 0; y < h; y += 20) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    const scale = w / 4.8; // world to screen
    const cartY = h - 40;
    const worldCenterX = w / 2;
    const screenCartX = worldCenterX + this.state.cartPosition * scale;

    // Track
    ctx.strokeStyle = "#1E1E2E";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, cartY);
    ctx.lineTo(w, cartY);
    ctx.stroke();

    // Track tick marks
    ctx.strokeStyle = "rgba(30, 30, 46, 0.5)";
    ctx.lineWidth = 1;
    for (let x = -2; x <= 2; x += 0.5) {
      const tx = worldCenterX + x * scale;
      ctx.beginPath();
      ctx.moveTo(tx, cartY - 5);
      ctx.lineTo(tx, cartY + 5);
      ctx.stroke();
    }

    // Cart (yellow rectangle)
    const cartW = 60;
    const cartH = 30;
    ctx.fillStyle = "#FACC15";
    ctx.fillRect(screenCartX - cartW / 2, cartY - cartH, cartW, cartH);

    // Pole (purple line)
    const poleLen = 80;
    const poleEndX = screenCartX + Math.sin(this.state.poleAngle) * poleLen;
    const poleEndY = cartY - cartH - Math.cos(this.state.poleAngle) * poleLen;

    ctx.strokeStyle = "#8B5CF6";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(screenCartX, cartY - cartH);
    ctx.lineTo(poleEndX, poleEndY);
    ctx.stroke();

    // Pole glow
    ctx.shadowColor = "#8B5CF6";
    ctx.shadowBlur = 8;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Pivot point
    ctx.fillStyle = "#8B5CF6";
    ctx.beginPath();
    ctx.arc(screenCartX, cartY - cartH, 5, 0, Math.PI * 2);
    ctx.fill();

    // Cart wheels
    ctx.fillStyle = "#9CA3AF";
    ctx.beginPath();
    ctx.arc(screenCartX - 20, cartY, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(screenCartX + 20, cartY, 6, 0, Math.PI * 2);
    ctx.fill();
  }
}
