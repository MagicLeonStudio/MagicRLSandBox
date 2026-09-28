export interface SnakeState {
  snake: [number, number][];
  food: [number, number];
  direction: number;
  steps: number;
  score: number;
}

export type SnakeAction = 0 | 1 | 2 | 3; // Up, Down, Left, Right

export const SNAKE_ACTIONS = ["Up", "Down", "Left", "Right"] as const;

export class SnakeEngine {
  gridSize = 10;
  state: SnakeState;
  maxSteps = 1000;

  constructor() {
    this.state = this.reset();
  }

  reset(): SnakeState {
    const startX = Math.floor(this.gridSize / 2);
    const startY = Math.floor(this.gridSize / 2);
    const snake: [number, number][] = [
      [startX, startY],
      [startX - 1, startY],
      [startX - 2, startY],
    ];

    let foodX = Math.floor(Math.random() * this.gridSize);
    let foodY = Math.floor(Math.random() * this.gridSize);
    while (snake.some(([sx, sy]) => sx === foodX && sy === foodY)) {
      foodX = Math.floor(Math.random() * this.gridSize);
      foodY = Math.floor(Math.random() * this.gridSize);
    }

    this.state = {
      snake,
      food: [foodX, foodY],
      direction: 3, // Right
      steps: 0,
      score: 0,
    };
    return this.state;
  }

  step(action: SnakeAction): {
    state: number[];
    reward: number;
    done: boolean;
    rawState: SnakeState;
  } {
    // Prevent 180-degree turns
    const opposites = [
      [0, 1], // Up opposite Down
      [1, 0], // Down opposite Up
      [2, 3], // Left opposite Right
      [3, 2], // Right opposite Left
    ];
    if (opposites[this.state.direction][0] !== action) {
      this.state.direction = action;
    }

    const head = this.state.snake[0];
    let newHead: [number, number] = [head[0], head[1]];

    switch (this.state.direction) {
      case 0: // Up
        newHead[1] -= 1;
        break;
      case 1: // Down
        newHead[1] += 1;
        break;
      case 2: // Left
        newHead[0] -= 1;
        break;
      case 3: // Right
        newHead[0] += 1;
        break;
    }

    this.state.steps++;

    // Wall collision
    if (
      newHead[0] < 0 ||
      newHead[0] >= this.gridSize ||
      newHead[1] < 0 ||
      newHead[1] >= this.gridSize
    ) {
      return {
        state: this.getObservation(),
        reward: -10,
        done: true,
        rawState: { ...this.state, snake: [...this.state.snake] },
      };
    }

    // Self collision (exclude tail which will move)
    if (this.state.snake.slice(0, -1).some(([sx, sy]) => sx === newHead[0] && sy === newHead[1])) {
      return {
        state: this.getObservation(),
        reward: -10,
        done: true,
        rawState: { ...this.state, snake: [...this.state.snake] },
      };
    }

    // Check food
    const ateFood = newHead[0] === this.state.food[0] && newHead[1] === this.state.food[1];

    this.state.snake.unshift(newHead);

    if (ateFood) {
      this.state.score++;
      // Spawn new food
      let foodX = Math.floor(Math.random() * this.gridSize);
      let foodY = Math.floor(Math.random() * this.gridSize);
      while (this.state.snake.some(([sx, sy]) => sx === foodX && sy === foodY)) {
        foodX = Math.floor(Math.random() * this.gridSize);
        foodY = Math.floor(Math.random() * this.gridSize);
      }
      this.state.food = [foodX, foodY];

      return {
        state: this.getObservation(),
        reward: 10,
        done: this.state.steps >= this.maxSteps,
        rawState: { ...this.state, snake: [...this.state.snake] },
      };
    } else {
      this.state.snake.pop();
    }

    // Direction reward: +0.1 if moving toward food, -0.1 if moving away
    const prevDist = Math.abs(head[0] - this.state.food[0]) + Math.abs(head[1] - this.state.food[1]);
    const newDist = Math.abs(newHead[0] - this.state.food[0]) + Math.abs(newHead[1] - this.state.food[1]);
    const directionReward = newDist < prevDist ? 0.1 : -0.1;

    return {
      state: this.getObservation(),
      reward: directionReward,
      done: this.state.steps >= this.maxSteps,
      rawState: { ...this.state, snake: [...this.state.snake] },
    };
  }

  getObservation(): number[] {
    const head = this.state.snake[0];
    const [hx, hy] = head;
    const [fx, fy] = this.state.food;
    const dir = this.state.direction;

    // Helper: check if a point is a wall or snake body (excluding tail which moves)
    const isDanger = (x: number, y: number): boolean => {
      if (x < 0 || x >= this.gridSize || y < 0 || y >= this.gridSize) return true;
      return this.state.snake.slice(0, -1).some(([sx, sy]) => sx === x && sy === y);
    };

    // Direction vectors: Up=0, Down=1, Left=2, Right=3
    const dirVecs: [number, number][] = [
      [0, -1], // Up
      [0, 1],  // Down
      [-1, 0], // Left
      [1, 0],  // Right
    ];

    // Right and left vectors relative to current direction
    // rightDirs[i] = direction to the right when facing direction i
    const rightDirs = [3, 2, 0, 1]; // Up->Right, Down->Left, Left->Up, Right->Down
    const leftDirs = [2, 3, 1, 0];  // Up->Left, Down->Right, Left->Down, Right->Up

    const straight = dirVecs[dir];
    const right = dirVecs[rightDirs[dir]];
    const left = dirVecs[leftDirs[dir]];

    const dangerStraight = isDanger(hx + straight[0], hy + straight[1]) ? 1 : 0;
    const dangerRight = isDanger(hx + right[0], hy + right[1]) ? 1 : 0;
    const dangerLeft = isDanger(hx + left[0], hy + left[1]) ? 1 : 0;

    // Current direction one-hot: [Up, Down, Left, Right]
    const dirOneHot = [
      dir === 0 ? 1 : 0,
      dir === 1 ? 1 : 0,
      dir === 2 ? 1 : 0,
      dir === 3 ? 1 : 0,
    ];

    // Food direction one-hot: [foodAbove, foodBelow, foodLeft, foodRight]
    const foodOneHot = [
      fy < hy ? 1 : 0,
      fy > hy ? 1 : 0,
      fx < hx ? 1 : 0,
      fx > hx ? 1 : 0,
    ];

    return [
      dangerStraight,
      dangerRight,
      dangerLeft,
      ...dirOneHot,
      ...foodOneHot,
    ];
  }

  getStateSize(): number {
    return 11;
  }

  getActionSize(): number {
    return 4;
  }

  render(canvas: HTMLCanvasElement): void {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const cellSize = canvas.width / this.gridSize;

    // Background
    ctx.fillStyle = "#0A0A12";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Grid lines
    ctx.strokeStyle = "#12121A";
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

    // Snake body
    for (let i = this.state.snake.length - 1; i >= 1; i--) {
      const [x, y] = this.state.snake[i];
      const t = i / Math.max(this.state.snake.length, 1);
      const r = Math.floor(124 + (76 - 124) * (1 - t));
      const g = Math.floor(58 + (29 - 58) * (1 - t));
      const b = Math.floor(237 + (149 - 237) * (1 - t));
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      const pad = 1;
      ctx.fillRect(
        x * cellSize + pad,
        y * cellSize + pad,
        cellSize - pad * 2,
        cellSize - pad * 2
      );
    }

    // Snake head (bright purple with glow)
    const [hx, hy] = this.state.snake[0];
    ctx.fillStyle = "#8B5CF6";
    const hPad = 1;
    ctx.fillRect(
      hx * cellSize + hPad,
      hy * cellSize + hPad,
      cellSize - hPad * 2,
      cellSize - hPad * 2
    );
    ctx.shadowColor = "#8B5CF6";
    ctx.shadowBlur = 12;
    ctx.fillRect(
      hx * cellSize + hPad,
      hy * cellSize + hPad,
      cellSize - hPad * 2,
      cellSize - hPad * 2
    );
    ctx.shadowBlur = 0;

    // Food (watermelon red circle)
    const [fx, fy] = this.state.food;
    const cx = fx * cellSize + cellSize / 2;
    const cy = fy * cellSize + cellSize / 2;
    const radius = cellSize * 0.35;

    ctx.fillStyle = "#FF6B6B";
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fill();

    ctx.shadowColor = "#FF6B6B";
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
  }
}
