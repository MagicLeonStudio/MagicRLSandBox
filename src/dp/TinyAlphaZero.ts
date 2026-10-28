import { NeuralNetwork } from "@/algorithms/NeuralNetwork";
import { mulberry32 } from "@/lib/rng";

export type Player = 1 | -1;
export type Board = Int8Array<ArrayBuffer>;

export const BOARD_SIZE = 4;
export const WIN_LENGTH = 3;
export const CELL_COUNT = BOARD_SIZE * BOARD_SIZE;

const LINES: number[][] = (() => {
  const lines: number[][] = [];
  for (let row = 0; row < BOARD_SIZE; row++) {
    for (let col = 0; col < BOARD_SIZE; col++) {
      const start = row * BOARD_SIZE + col;
      if (col <= BOARD_SIZE - WIN_LENGTH) lines.push(Array.from({ length: WIN_LENGTH }, (_, k) => start + k));
      if (row <= BOARD_SIZE - WIN_LENGTH) lines.push(Array.from({ length: WIN_LENGTH }, (_, k) => start + k * BOARD_SIZE));
      if (row <= BOARD_SIZE - WIN_LENGTH && col <= BOARD_SIZE - WIN_LENGTH) lines.push(Array.from({ length: WIN_LENGTH }, (_, k) => start + k * (BOARD_SIZE + 1)));
      if (row <= BOARD_SIZE - WIN_LENGTH && col >= WIN_LENGTH - 1) lines.push(Array.from({ length: WIN_LENGTH }, (_, k) => start + k * (BOARD_SIZE - 1)));
    }
  }
  return lines;
})();

export interface PVOutput {
  probs: number[];
  value: number;
}

export interface TrainSample {
  board: Board;
  player: Player;
  pi: number[];
  z: number;
}

export interface MCTSActionStat {
  action: number;
  visits: number;
  meanValue: number;
  prior: number;
  score: number;
}

export interface EvalResult {
  win: number;
  draw: number;
  loss: number;
}

export function legalActions(board: Board): number[] {
  const actions: number[] = [];
  for (let i = 0; i < CELL_COUNT; i++) if (board[i] === 0) actions.push(i);
  return actions;
}

export function winner(board: Board): Player | 0 | null {
  for (const [a, b, c] of LINES) {
    const v = board[a];
    if ((v === 1 || v === -1) && v === board[b] && board[b] === board[c]) return v as Player;
  }
  if (legalActions(board).length > 0) return null;
  return 0;
}

export function applyAction(board: Board, action: number, player: Player): Board {
  const next = new Int8Array(board);
  next[action] = player;
  return next;
}

export function boardKey(board: Board, player: Player): string {
  return `${Array.from(board).join("")}:${player}`;
}

/** Shared-trunk policy/value network. The solver is never used for training. */
export class PolicyValueNet {
  nn: NeuralNetwork;

  constructor(seed = 1) {
    this.nn = new NeuralNetwork([CELL_COUNT * 2, 32, 32, CELL_COUNT + 1], seed);
  }

  static input(board: Board, player: Player): number[] {
    const input: number[] = [];
    for (let i = 0; i < CELL_COUNT; i++) input.push(board[i] === player ? 1 : 0);
    for (let i = 0; i < CELL_COUNT; i++) input.push(board[i] === -player ? 1 : 0);
    return input;
  }

  forward(board: Board, player: Player): PVOutput {
    const raw = this.nn.forward(PolicyValueNet.input(board, player));
    const exps = raw.slice(0, CELL_COUNT).map((v) => Math.exp(Math.min(v, 30)));
    const sum = exps.reduce((a, b) => a + b, 0);
    return {
      probs: exps.map((v) => v / sum),
      value: Math.tanh(raw[CELL_COUNT]),
    };
  }

  train(sample: TrainSample, lr: number): number {
    const out = this.forward(sample.board, sample.player);
    const grad = out.probs.map((p, i) => p - sample.pi[i]);
    grad.push((out.value - sample.z) * (1 - out.value * out.value));
    this.nn.backwardOutputGradient(grad, lr);
    const policyLoss = -sample.pi.reduce((sum, p, i) => sum + (p > 0 ? p * Math.log(Math.max(out.probs[i], 1e-9)) : 0), 0);
    return policyLoss + 0.5 * (out.value - sample.z) ** 2;
  }
}

interface Node {
  board: Board;
  player: Player;
  parent: Node | null;
  action: number;
  children: Node[];
  visits: number;
  valueSum: number;
  prior: number;
  expanded: boolean;
}

function createNode(board: Board, player: Player, parent: Node | null, action: number, prior: number): Node {
  return { board, player, parent, action, children: [], visits: 0, valueSum: 0, prior, expanded: false };
}

function expand(node: Node, net: PolicyValueNet) {
  const output = net.forward(node.board, node.player);
  const legal = legalActions(node.board);
  const priorSum = legal.reduce((sum, action) => sum + output.probs[action], 0);
  for (const action of legal) {
    const prior = priorSum > 0 ? output.probs[action] / priorSum : 1 / legal.length;
    node.children.push(createNode(applyAction(node.board, action, node.player), -node.player as Player, node, action, prior));
  }
  node.expanded = true;
}

function bestChild(node: Node, cPuct: number): Node {
  const parentVisits = Math.sqrt(Math.max(1, node.visits));
  let best = node.children[0];
  let bestScore = -Infinity;
  for (const child of node.children) {
    const mean = child.visits > 0 ? child.valueSum / child.visits : 0;
    const score = -mean + cPuct * child.prior * parentVisits / (1 + child.visits);
    if (score > bestScore) {
      bestScore = score;
      best = child;
    }
  }
  return best;
}

/** PUCT search with a policy prior and a value head, without random rollouts. */
export function runMCTS(board: Board, player: Player, net: PolicyValueNet, simulations = 50, cPuct = 1.5): MCTSActionStat[] {
  const root = createNode(new Int8Array(board), player, null, -1, 1);
  for (let i = 0; i < simulations; i++) {
    let node = root;
    const path: Node[] = [node];
    while (node.expanded && node.children.length > 0 && winner(node.board) === null) {
      node = bestChild(node, cPuct);
      path.push(node);
    }
    const terminal = winner(node.board);
    let value = 0;
    if (terminal !== null) {
      value = terminal === 0 ? 0 : terminal === node.player ? 1 : -1;
    } else {
      if (!node.expanded) expand(node, net);
      value = net.forward(node.board, node.player).value;
    }
    for (const ancestor of path) {
      ancestor.visits++;
      ancestor.valueSum += ancestor.player === node.player ? value : -value;
    }
  }
  if (!root.expanded) expand(root, net);
  return root.children.map((child) => {
    const meanValue = child.visits > 0 ? -child.valueSum / child.visits : 0;
    return {
      action: child.action,
      visits: child.visits,
      meanValue,
      prior: child.prior,
      score: meanValue + cPuct * child.prior * Math.sqrt(Math.max(1, root.visits)) / (1 + child.visits),
    };
  }).sort((a, b) => b.visits - a.visits);
}

export function mctsAction(board: Board, player: Player, net: PolicyValueNet, simulations = 50, sample = false, rng: () => number = Math.random): number {
  const stats = runMCTS(board, player, net, simulations);
  if (!sample) return stats[0].action;
  const total = stats.reduce((sum, stat) => sum + stat.visits, 0);
  if (total === 0) {
    const legal = legalActions(board);
    return legal[Math.floor(rng() * legal.length)];
  }
  let threshold = rng() * total;
  for (const stat of stats) {
    threshold -= stat.visits;
    if (threshold <= 0) return stat.action;
  }
  return stats[stats.length - 1].action;
}

/** Exact minimax, used only as an external evaluator. */
export class MiniMaxSolver {
  private memo = new Map<string, { score: number; action: number }>();

  solve(board: Board, player: Player, rng?: () => number): { score: number; action: number } {
    const result = this.search(board, player, -Infinity, Infinity);
    if (!rng) return result;
    const alternatives = this.bestActions(board, player, result.score);
    return { score: result.score, action: alternatives[Math.floor(rng() * alternatives.length)] };
  }

  private search(board: Board, player: Player, alpha: number, beta: number): { score: number; action: number } {
    const key = boardKey(board, player);
    const cached = this.memo.get(key);
    if (cached) return cached;
    const terminal = winner(board);
    if (terminal !== null) {
      const result = { score: terminal === 0 ? 0 : terminal === player ? 1 : -1, action: -1 };
      this.memo.set(key, result);
      return result;
    }
    let bestScore = -Infinity;
    let bestAction = -1;
    for (const action of this.orderedActions(board)) {
      const child = this.search(applyAction(board, action, player), -player as Player, -beta, -alpha);
      const score = -child.score;
      if (score > bestScore) {
        bestScore = score;
        bestAction = action;
      }
      alpha = Math.max(alpha, bestScore);
      if (alpha >= beta) break;
    }
    const result = { score: bestScore, action: bestAction };
    this.memo.set(key, result);
    return result;
  }

  private orderedActions(board: Board): number[] {
    const legal = legalActions(board);
    const center = [5, 6, 9, 10];
    return [
      ...center.filter((action) => legal.includes(action)),
      ...legal.filter((action) => !center.includes(action)),
    ];
  }
  private bestActions(board: Board, player: Player, bestScore: number): number[] {
    const actions: number[] = [];
    for (const action of legalActions(board)) {
      const score = -this.solve(applyAction(board, action, player), -player as Player).score;
      if (score === bestScore) actions.push(action);
    }
    return actions;
  }
}

export function evaluateAgainstSolver(net: PolicyValueNet, solver: MiniMaxSolver, games = 12, simulations = 0, seed = 42): EvalResult {
  const rng = mulberry32(seed);
  const result: EvalResult = { win: 0, draw: 0, loss: 0 };
  for (let game = 0; game < games; game++) {
    let board = new Int8Array(CELL_COUNT);
    let player: Player = 1;
    const netPlayer: Player = game % 2 === 0 ? 1 : -1;
    while (winner(board) === null) {
      if (player === netPlayer) {
        const action = simulations > 0 ? mctsAction(board, player, net, simulations) : (() => {
          const output = net.forward(board, player);
          return legalActions(board).reduce((best, action) => output.probs[action] > output.probs[best] ? action : best, legalActions(board)[0]);
        })();
        board = applyAction(board, action, player);
      } else {
        board = applyAction(board, solver.solve(board, player, rng).action, player);
      }
      player = -player as Player;
    }
    const outcome = winner(board);
    if (outcome === 0) result.draw++;
    else if (outcome === netPlayer) result.win++;
    else result.loss++;
  }
  return result;
}

export function selfPlayIteration(net: PolicyValueNet, options: { games?: number; simulations?: number; epochs?: number; seed?: number }): { samples: number; loss: number } {
  const games = options.games ?? 24;
  const simulations = options.simulations ?? 40;
  const epochs = options.epochs ?? 4;
  const rng = mulberry32(options.seed ?? 1);
  const samples: TrainSample[] = [];
  for (let game = 0; game < games; game++) {
    let board = new Int8Array(CELL_COUNT);
    let player: Player = 1;
    const trajectory: { board: Board; player: Player; pi: number[] }[] = [];
    while (winner(board) === null) {
      const stats = runMCTS(board, player, net, simulations);
      const total = stats.reduce((sum, stat) => sum + stat.visits, 0);
      const pi = Array.from({ length: CELL_COUNT }, (_, action) => {
        const stat = stats.find((item) => item.action === action);
        return total > 0 && stat ? stat.visits / total : 0;
      });
      trajectory.push({ board: new Int8Array(board), player, pi });
      const totalVisits = total;
      let threshold = rng() * totalVisits;
      let action = stats[stats.length - 1].action;
      for (const stat of stats) {
        threshold -= stat.visits;
        if (threshold <= 0) { action = stat.action; break; }
      }
      board = applyAction(board, action, player);
      player = -player as Player;
    }
    const outcome = winner(board);
    for (const state of trajectory) {
      const z = outcome === 0 ? 0 : outcome === state.player ? 1 : -1;
      samples.push({ board: state.board, player: state.player, pi: state.pi, z });
    }
  }
  let loss = 0;
  for (let epoch = 0; epoch < epochs; epoch++) {
    for (const sample of samples) loss += net.train(sample, 0.001);
  }
  return { samples: samples.length, loss: loss / Math.max(1, samples.length * epochs) };
}
