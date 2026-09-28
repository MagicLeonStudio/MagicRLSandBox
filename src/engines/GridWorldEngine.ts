export interface GridWorldState {
  agentX: number;
  agentY: number;
  targetX: number;
  targetY: number;
  obstacles: [number, number][];
  steps: number;
}

export type GridAction = 0 | 1 | 2 | 3; // Up, Down, Left, Right

export const GRID_ACTIONS = ["Up", "Down", "Left", "Right"] as const;

export class GridWorldEngine {
  gridSize = 5;
  state: GridWorldState;
  maxSteps = 100;

  constructor() {
    this.state = this.reset();
  }

  reset(): GridWorldState {
    const agentX = 0;
    const agentY = 0;

    // Place target at random position excluding agent start
    let targetX = Math.floor(Math.random() * this.gridSize);
    let targetY = Math.floor(Math.random() * this.gridSize);
    while (targetX === agentX && targetY === agentY) {
      targetX = Math.floor(Math.random() * this.gridSize);
      targetY = Math.floor(Math.random() * this.gridSize);
    }

    // Place 1-2 obstacles randomly
    const obstacleCount = 1 + Math.floor(Math.random() * 2);
    const obstacles: [number, number][] = [];
    for (let i = 0; i < obstacleCount; i++) {
      let ox = Math.floor(Math.random() * this.gridSize);
      let oy = Math.floor(Math.random() * this.gridSize);
      while (
        (ox === agentX && oy === agentY) ||
        (ox === targetX && oy === targetY) ||
        obstacles.some(([x, y]) => x === ox && y === oy)
      ) {
        ox = Math.floor(Math.random() * this.gridSize);
        oy = Math.floor(Math.random() * this.gridSize);
      }
      obstacles.push([ox, oy]);
    }

    this.state = {
      agentX,
      agentY,
      targetX,
      targetY,
      obstacles,
      steps: 0,
    };
    return this.state;
  }

  step(action: GridAction): {
    state: number[];
    reward: number;
    done: boolean;
    rawState: GridWorldState;
  } {
    let { agentX, agentY } = this.state;

    // Execute action
    switch (action) {
      case 0: // Up
        agentY = Math.max(0, agentY - 1);
        break;
      case 1: // Down
        agentY = Math.min(this.gridSize - 1, agentY + 1);
        break;
      case 2: // Left
        agentX = Math.max(0, agentX - 1);
        break;
      case 3: // Right
        agentX = Math.min(this.gridSize - 1, agentX + 1);
        break;
    }

    this.state.agentX = agentX;
    this.state.agentY = agentY;
    this.state.steps++;

    let reward = -0.1; // Step penalty
    let done = false;

    // Check target reached
    if (agentX === this.state.targetX && agentY === this.state.targetY) {
      reward = 10;
      done = true;
    }

    // Check obstacle collision
    if (this.state.obstacles.some(([ox, oy]) => ox === agentX && oy === agentY)) {
      reward = -5;
      done = true;
    }

    // Check max steps
    if (this.state.steps >= this.maxSteps) {
      done = true;
    }

    return {
      state: this.getObservation(),
      reward,
      done,
      rawState: { ...this.state },
    };
  }

  getObservation(): number[] {
    return [
      this.state.agentX,
      this.state.agentY,
      this.state.targetX,
      this.state.targetY,
    ];
  }

  getStateSize(): number {
    return 4;
  }

  getActionSize(): number {
    return 4;
  }

  render(canvas: HTMLCanvasElement): void {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const cellSize = canvas.width / this.gridSize;
    const { agentX, agentY, targetX, targetY, obstacles } = this.state;

    // Background
    ctx.fillStyle = "#0D0D15";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Grid lines
    ctx.strokeStyle = "#1E1E2E";
    ctx.lineWidth = 1;
    for (let i = 0; i <= this.gridSize; i++) {
      ctx.beginPath();
      ctx.moveTo(i * cellSize, 0);
      ctx.lineTo(i * cellSize, canvas.height);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, i * cellSize);
      ctx.lineTo(canvas.width, i * cellSize);
      ctx.stroke();
    }

    // Coordinate labels
    ctx.fillStyle = "rgba(107, 114, 128, 0.2)";
    ctx.font = "8px 'JetBrains Mono', monospace";
    for (let y = 0; y < this.gridSize; y++) {
      for (let x = 0; x < this.gridSize; x++) {
        ctx.fillText(`(${x},${y})`, x * cellSize + 4, y * cellSize + 12);
      }
    }

    // Obstacles (red)
    for (const [ox, oy] of obstacles) {
      const pad = cellSize * 0.1;
      ctx.fillStyle = "#FF6B6B";
      ctx.fillRect(
        ox * cellSize + pad,
        oy * cellSize + pad,
        cellSize - pad * 2,
        cellSize - pad * 2
      );
    }

    // Target (yellow)
    const tPad = cellSize * 0.1;
    ctx.fillStyle = "#FACC15";
    ctx.fillRect(
      targetX * cellSize + tPad,
      targetY * cellSize + tPad,
      cellSize - tPad * 2,
      cellSize - tPad * 2
    );

    // Agent (purple)
    const aPad = cellSize * 0.15;
    ctx.fillStyle = "#8B5CF6";
    ctx.fillRect(
      agentX * cellSize + aPad,
      agentY * cellSize + aPad,
      cellSize - aPad * 2,
      cellSize - aPad * 2
    );

    // Agent glow
    ctx.shadowColor = "#8B5CF6";
    ctx.shadowBlur = 12;
    ctx.fillRect(
      agentX * cellSize + aPad,
      agentY * cellSize + aPad,
      cellSize - aPad * 2,
      cellSize - aPad * 2
    );
    ctx.shadowBlur = 0;
  }
}
