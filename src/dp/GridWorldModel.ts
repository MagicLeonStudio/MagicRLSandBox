/**
 * GridWorldModel —— GridWorld 的环境模型（已知转移与奖励）
 * 与 engines/GridWorldEngine 规则保持一致：动作 0=上 1=下 2=左 3=右，撞墙停留。
 * 动态规划类算法（Value/Policy Iteration）假设已知模型，故直接消费本类。
 */
export class GridWorldModel {
  readonly size: number;
  readonly goal: number;
  readonly walls: Set<number>;
  readonly stepPenalty: number;
  readonly obstaclePenalty?: number;

  constructor(size = 5, walls: number[] = [], stepPenalty = -0.01, obstaclePenalty?: number) {
    this.size = size;
    this.goal = size * size - 1;
    this.walls = new Set(walls);
    this.stepPenalty = stepPenalty;
    this.obstaclePenalty = obstaclePenalty;
  }

  get stateCount(): number {
    return this.size * this.size;
  }

  isWall(s: number): boolean {
    return this.walls.has(s);
  }

  isTerminal(s: number): boolean {
    return s === this.goal;
  }

  /** 无效（墙或终止）状态返回空数组 */
  actions(s: number): number[] {
    if (this.isWall(s) || this.isTerminal(s)) return [];
    return [0, 1, 2, 3];
  }

  /** 确定性转移：返回执行动作后的下一状态 */
  nextState(s: number, a: number): number {
    const row = Math.floor(s / this.size);
    const col = s % this.size;
    let r = row, c = col;
    if (a === 0) r = Math.max(0, row - 1);
    else if (a === 1) r = Math.min(this.size - 1, row + 1);
    else if (a === 2) c = Math.max(0, col - 1);
    else if (a === 3) c = Math.min(this.size - 1, col + 1);
    const ns = r * this.size + c;
    return this.walls.has(ns) ? s : ns; // 撞墙（含障碍）则原地不动
  }

  reward(s: number, a: number, ns: number): number {
    if (this.isTerminal(ns)) return 10;
    // obstacle bump: intended cell is in-bounds but blocked; boundary bump keeps stepPenalty
    if (ns === s && this.obstaclePenalty !== undefined) {
      const row = Math.floor(s / this.size);
      const col = s % this.size;
      let r = row, c = col;
      if (a === 0) r = row - 1;
      else if (a === 1) r = row + 1;
      else if (a === 2) c = col - 1;
      else c = col + 1;
      const inBounds = r >= 0 && r < this.size && c >= 0 && c < this.size;
      if (inBounds && this.walls.has(r * this.size + c)) return this.obstaclePenalty;
    }
    return this.stepPenalty;
  }
}
