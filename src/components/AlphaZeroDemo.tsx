import { useCallback, useEffect, useRef, useState, Fragment } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { Play, Pause, StepForward, RotateCcw, Search, BrainCircuit, Trophy } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import TeX from "@/components/TeX";
import { PolicyValueNet, MiniMaxSolver, runMCTS, mctsAction, legalActions, applyAction, SelfPlaySession, evaluateAgainstSolver, winner, CELL_COUNT, BOARD_SIZE } from "@/dp/TinyAlphaZero";
import type { MCTSActionStat, PVOutput, Player, Board } from "@/dp/TinyAlphaZero";

const GOLD = "#FFD21F";
const PURPLE = "#8B5CF6";
const BLUE = "#3EC6F0";
const MELON = "#FF4D6D";

interface EvalHistory {
  iteration: number;
  samples: number;
  loss: number;
  policyWin: number;
  policyLoss: number;
  mctsWin: number;
  mctsLoss: number;
}

function cellName(index: number): string {
  return `${String.fromCharCode(65 + (index % 5))}${Math.floor(index / 5) + 1}`;
}

function BoardGrid({
  board,
  onCellClick,
  values,
  visits,
  compact = false,
}: {
  board: Board;
  onCellClick?: (index: number) => void;
  values?: number[];
  visits?: number[];
  compact?: boolean;
}) {
  return (
    <div className="w-fit rounded-lg border border-[#2A2445] bg-[#07060E] p-2">
      <div className="grid w-fit gap-1.5" style={{ gridTemplateColumns: `12px repeat(${BOARD_SIZE}, auto)` }}>
        {/* column letters header (corner cell empty) */}
        <span />
        {Array.from({ length: BOARD_SIZE }, (_, col) => (
          <span key={`col-${col}`} className="self-end text-center font-mono text-[9px] text-[#6B6390]">
            {String.fromCharCode(65 + col)}
          </span>
        ))}
        {Array.from({ length: BOARD_SIZE }, (_, row) => (
          <Fragment key={`row-${row}`}>
            <span className="flex items-center justify-center font-mono text-[9px] text-[#6B6390]">{row + 1}</span>
            {Array.from({ length: BOARD_SIZE }, (_, col) => {
              const index = row * BOARD_SIZE + col;
              const stone = board[index];
              const value = values?.[index];
              const visit = visits?.[index];
              return (
                <button
                  key={index}
                  title={cellName(index)}
                  disabled={stone !== 0 || !onCellClick}
                  onClick={() => onCellClick?.(index)}
                  className={`relative flex items-center justify-center rounded-md border font-bold transition-colors disabled:cursor-default ${compact ? "h-8 w-8 text-xs md:h-9 md:w-9" : "h-11 w-11 text-sm md:h-14 md:w-14 md:text-base"}`}
                  style={{
                    borderColor: stone === 1 ? GOLD : stone === -1 ? PURPLE : "#2A2445",
                    backgroundColor: value !== undefined ? `rgba(255,210,31,${Math.min(0.55, value * 0.55)})` : stone === 0 ? "#0D0B18" : "#141126",
                  }}
                >
                  <span style={{ color: stone === 1 ? GOLD : stone === -1 ? PURPLE : "#6B6390" }}>
                    {stone === 1 ? "×" : stone === -1 ? "○" : ""}
                  </span>
                  {value !== undefined && <span className="absolute bottom-0.5 right-1 font-mono text-[9px] text-[#F1EEFB]">{(value * 100).toFixed(0)}%</span>}
                  {visit !== undefined && visit > 0 && <span className="absolute bottom-0.5 right-1 font-mono text-[9px] text-[#F1EEFB]">{visit}</span>}
                </button>
              );
            })}
          </Fragment>
        ))}
      </div>
    </div>
  );
}

function Metric({ label, value, accent = "#F1EEFB" }: { label: string; value: string; accent?: string }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-[0.12em] text-[#6B6390]">{label}</div>
      <div className="font-mono text-sm font-semibold" style={{ color: accent }}>{value}</div>
    </div>
  );
}

function ActionStatsTable({ stats, emptyHint }: { stats: MCTSActionStat[]; emptyHint: string }) {
  const totalVisits = stats.reduce((sum, stat) => sum + stat.visits, 0);
  return (
    <div className="overflow-x-auto">
      <div className="mb-2 flex items-center justify-between text-[10px] text-[#A79FC7]">
        <span>根节点动作统计</span>
        <span className="font-mono">总访问 {totalVisits}</span>
      </div>
      <table className="w-full min-w-[520px] border-collapse text-[11px]">
        <thead>
          <tr className="border-b border-[#2A2445] text-left text-[#6B6390]">
            <th className="py-1.5 pr-3 font-normal">动作</th>
            <th className="py-1.5 pr-3 font-normal">访问 N</th>
            <th className="py-1.5 pr-3 font-normal">均值 Q</th>
            <th className="py-1.5 pr-3 font-normal">先验 P</th>
            <th className="py-1.5 pr-3 font-normal">PUCT</th>
          </tr>
        </thead>
        <tbody>
          {stats.map((stat, index) => (
            <tr key={stat.action} className="border-b border-[#2A2445]/60" style={{ backgroundColor: index === 0 ? "rgba(255,210,31,0.06)" : undefined }}>
              <td className="py-1.5 pr-3 font-mono" style={{ color: index === 0 ? GOLD : "#A79FC7" }}>{cellName(stat.action)}</td>
              <td className="py-1.5 pr-3 font-mono text-[#F1EEFB]">{stat.visits}</td>
              <td className="py-1.5 pr-3 font-mono text-[#A79FC7]">{stat.meanValue.toFixed(3)}</td>
              <td className="py-1.5 pr-3 font-mono text-[#A79FC7]">{stat.prior.toFixed(3)}</td>
              <td className="py-1.5 pr-3 font-mono" style={{ color: index === 0 ? GOLD : "#A79FC7" }}>{stat.score.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {stats.length === 0 && <div className="py-6 text-center text-xs text-[#6B6390]">{emptyHint}</div>}
    </div>
  );
}

export default function AlphaZeroDemo() {
  const [tab, setTab] = useState<"search" | "train" | "play">("search");
  const netRef = useRef<PolicyValueNet | null>(null);
  const solverRef = useRef<MiniMaxSolver | null>(null);
  if (!netRef.current) netRef.current = new PolicyValueNet(7);
  if (!solverRef.current) solverRef.current = new MiniMaxSolver();
  const net = netRef.current;
  const solver = solverRef.current;

  const [board, setBoard] = useState<Board>(new Int8Array(CELL_COUNT));
  const [player, setPlayer] = useState<Player>(1);
  const [stats, setStats] = useState<MCTSActionStat[]>([]);
  const [pv, setPv] = useState<PVOutput>(() => net.forward(new Int8Array(CELL_COUNT), 1));
  const [simulations, setSimulations] = useState(100);
  const [cPuct, setCPuct] = useState(1.5);

  const runSearch = useCallback(() => {
    if (outcome !== null) {
      setStats([]);
      return;
    }
    const nextStats = runMCTS(board, player, net, simulations, cPuct);
    setStats(nextStats);
    setPv(net.forward(board, player));
  }, [board, player, net, simulations, cPuct]);

  useEffect(() => { runSearch(); }, []);

  const makeMove = (index: number) => {
    if (board[index] !== 0 || outcome !== null) return;
    const next = new Int8Array(board);
    next[index] = player;
    setBoard(next);
    setPlayer(-player as Player);
    setPv(net.forward(next, -player as Player));
    setStats([]);
  };

  const resetBoard = () => {
    setBoard(new Int8Array(CELL_COUNT));
    setPlayer(1);
    setPv(net.forward(new Int8Array(CELL_COUNT), 1));
    setStats([]);
  };

  const [iteration, setIteration] = useState(0);
  const [samples, setSamples] = useState(0);
  const [loss, setLoss] = useState(0);
  const [history, setHistory] = useState<EvalHistory[]>([]);
  const [playing, setPlaying] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [latestEval, setLatestEval] = useState<{ policyWin: number; policyLoss: number; mctsWin: number; mctsLoss: number }>({ policyWin: 0, policyLoss: 0, mctsWin: 0, mctsLoss: 0 });
  const iterationRef = useRef(0);
  const evaluatedRef = useRef(false);

  const runEvaluation = useCallback(() => {
    solver.maxDepth = 6;
    const policy = evaluateAgainstSolver(net, solver, 12, 0, 42);
    const mcts = evaluateAgainstSolver(net, solver, 6, 100, 42);
    const point = { policyWin: policy.win, policyLoss: policy.loss, mctsWin: mcts.win, mctsLoss: mcts.loss };
    setLatestEval(point);
    return point;
  }, [net, solver]);

  // 深度受限 minimax 评估很重（5x5 深度 6），不进训练页不付这笔账；
  // 首次切到训练 Tab 时用 setTimeout 让“评估中”提示先渲染出来
  const ensureEvaluated = useCallback(() => {
    if (evaluatedRef.current) return;
    evaluatedRef.current = true;
    setEvaluating(true);
    setTimeout(() => {
      runEvaluation();
      setEvaluating(false);
    }, 30);
  }, [runEvaluation]);

  // 实时自我对弈：session 按“一手一步”推进，棋盘与根节点统计随每一手刷新
  const sessionRef = useRef<SelfPlaySession | null>(null);
  const movesPerTickRef = useRef(5);
  const [movesPerTick, setMovesPerTick] = useState(5);
  const [liveBoard, setLiveBoard] = useState<Board>(new Int8Array(CELL_COUNT));
  const [livePlayer, setLivePlayer] = useState<Player>(1);
  const [liveGames, setLiveGames] = useState(0);
  const [liveStats, setLiveStats] = useState<MCTSActionStat[]>([]);

  const advance = useCallback((moves: number) => {
    if (iterationRef.current >= 100) { setPlaying(false); return; }
    let session = sessionRef.current;
    if (!session) {
      session = new SelfPlaySession(net, { games: 24, simulations: 300, seed: iterationRef.current + 1 });
      sessionRef.current = session;
    }
    let lastStats: MCTSActionStat[] | null = null;
    for (let i = 0; i < moves; i++) {
      const stats = session.step();
      if (stats) lastStats = stats;
      if (session.done) break;
    }
    setLiveBoard(new Int8Array(session.board));
    setLivePlayer(session.player);
    setLiveGames(session.gamesDone);
    if (lastStats) setLiveStats(lastStats);
    if (session.done) {
      const meanLoss = session.train(10);
      const next = iterationRef.current + 1;
      iterationRef.current = next;
      setIteration(next);
      setSamples(session.samples.length);
      setLoss(meanLoss);
      let evaluation = latestEval;
      if (next % 10 === 0 || next === 100) evaluation = runEvaluation();
      const point: EvalHistory = {
        iteration: next,
        samples: session.samples.length,
        loss: meanLoss,
        policyWin: evaluation.policyWin,
        policyLoss: evaluation.policyLoss,
        mctsWin: evaluation.mctsWin,
        mctsLoss: evaluation.mctsLoss,
      };
      setHistory((prev) => [...prev, point]);
      sessionRef.current = null;
      if (next >= 100) setPlaying(false);
    }
  }, [net, latestEval, runEvaluation]);

  const trainTick = useCallback(() => advance(movesPerTickRef.current), [advance]);

  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(trainTick, 150);
    return () => clearInterval(timer);
  }, [playing, trainTick]);

  const resetTraining = () => {
    setPlaying(false);
    netRef.current = new PolicyValueNet(7);
    iterationRef.current = 0;
    sessionRef.current = null;
    setIteration(0);
    setSamples(0);
    setLoss(0);
    setHistory([]);
    setLiveBoard(new Int8Array(CELL_COUNT));
    setLivePlayer(1);
    setLiveGames(0);
    setLiveStats([]);
    setEvaluating(true);
    setTimeout(() => {
      runEvaluation();
      setEvaluating(false);
    }, 30);
  };

  const totalVisits = stats.reduce((sum, stat) => sum + stat.visits, 0);
  const visitDist = Array.from({ length: CELL_COUNT }, (_, action) => {
    const stat = stats.find((item) => item.action === action);
    return totalVisits > 0 && stat ? stat.visits / totalVisits : 0;
  });
  const liveVisits = Array.from({ length: CELL_COUNT }, (_, action) => liveStats.find((item) => item.action === action)?.visits ?? 0);

  // ---- 人机对战：对手是内存中当前的网络（随训练实时变强） ----
  const [playBoard, setPlayBoard] = useState<Board>(new Int8Array(CELL_COUNT));
  const [playTurn, setPlayTurn] = useState<Player>(1);
  const [humanSide, setHumanSide] = useState<Player>(1);
  const [agentMode, setAgentMode] = useState<"net" | "mcts">("mcts");
  const [thinking, setThinking] = useState(false);
  const [score, setScore] = useState({ win: 0, draw: 0, loss: 0 });
  const playOutcome = winner(playBoard);

  const agentThink = useCallback((b: Board, p: Player): number => {
    if (agentMode === "mcts") return mctsAction(b, p, net, 100);
    const output = net.forward(b, p);
    return legalActions(b).reduce((best, a) => (output.probs[a] > output.probs[best] ? a : best), legalActions(b)[0]);
  }, [agentMode, net]);

  const resetPlay = useCallback((side: Player) => {
    setPlayBoard(new Int8Array(CELL_COUNT));
    setPlayTurn(1);
    setHumanSide(side);
    setThinking(false);
  }, []);

  // 轮到智能体时延迟落子（留出“思考中”的感知时间）
  useEffect(() => {
    if (tab !== "play" || playOutcome !== null || playTurn === humanSide) return;
    setThinking(true);
    const timer = setTimeout(() => {
      const action = agentThink(playBoard, playTurn);
      setPlayBoard(applyAction(playBoard, action, playTurn));
      setPlayTurn(humanSide);
      setThinking(false);
    }, 450);
    return () => clearTimeout(timer);
  }, [tab, playBoard, playTurn, humanSide, playOutcome, agentThink]);

  // 终局记分（棋盘停止变化后只触发一次）
  useEffect(() => {
    if (playOutcome === null) return;
    setScore((s) => (playOutcome === 0 ? { ...s, draw: s.draw + 1 } : playOutcome === humanSide ? { ...s, win: s.win + 1 } : { ...s, loss: s.loss + 1 }));
  }, [playOutcome, humanSide]);

  const playMove = (index: number) => {
    if (playBoard[index] !== 0 || playOutcome !== null || thinking || playTurn !== humanSide) return;
    setPlayBoard(applyAction(playBoard, index, humanSide));
    setPlayTurn(-humanSide as Player);
  };

  const outcome = winner(board);

  return (
    <div id="alphazero-demo" className="overflow-hidden rounded-xl border border-[#2A2445] bg-[#07060E] text-[#F1EEFB] shadow-[0_0_24px_rgba(139,92,246,0.12)]">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#2A2445] bg-[#0D0B18] px-4 py-3">
        <div>
          <div className="font-mono text-[10px] tracking-[0.18em] text-[#8B5CF6]">TINY ALPHAZERO · SEARCH AS POLICY IMPROVEMENT</div>
          <div className="mt-1 text-sm font-semibold">5×5 三子连线上的自我对弈与策略蒸馏</div>
        </div>
        <div className="hidden md:block text-xs text-[#A79FC7]">
          <TeX math="a^*=\arg\max_a\left[Q(s,a)+cP(s,a)\frac{\sqrt{N(s)}}{1+N(s,a)}\right]" />
        </div>
      </div>

      <div className="border-b border-[#2A2445] p-3">
        <div className="flex rounded-lg border border-[#2A2445] p-1 text-[11px]">
          {(["search", "train", "play"] as const).map((item) => (
            <button
              key={item}
              onClick={() => { setTab(item); if (item === "train") ensureEvaluated(); }}
              className="flex-1 rounded-md px-3 py-2"
              style={{ backgroundColor: tab === item ? "#1C1832" : "transparent", color: tab === item ? GOLD : "#A79FC7" }}
            >
              {item === "search" ? "① PUCT 搜索与策略蒸馏" : item === "train" ? "② 自我对弈训练与强度评估" : "③ 人机对战"}
            </button>
          ))}
        </div>
      </div>

      {tab === "search" ? (
        <div className="space-y-4 p-4">
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-[auto_minmax(0,1fr)]">
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <BoardGrid board={board} onCellClick={makeMove} />
                <div className="max-w-[250px] rounded-lg border border-[#2A2445] bg-[#0D0B18] p-3 text-[11px] leading-relaxed text-[#A79FC7]">
                  <div className="mb-2 text-xs font-semibold text-[#F1EEFB]">规则与评估</div>
                  <div>双方轮流落子，横、竖、斜任意一条连续三子连线即胜。</div>
                  <div>棋盘满仍未分胜负为平局。</div>
                  <div className="pt-1 text-[#6B6390]">深度受限 minimax 只用于赛后评估，训练只使用自我对弈。当前：<b style={{ color: player === 1 ? GOLD : PURPLE }}>{player === 1 ? "× 先手" : "○ 后手"}</b></div>
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Button size="sm" className="bg-[#FFD21F] text-black hover:bg-[#FFE873]" onClick={runSearch}>
                  <Search className="mr-1 h-4 w-4" />运行 MCTS
                </Button>
                <Button size="sm" variant="outline" onClick={resetBoard}><RotateCcw className="mr-1 h-4 w-4" />重置棋局</Button>
              </div>
              <div className="space-y-3">
                <div>
                  <div className="mb-1 flex justify-between text-[10px] text-[#A79FC7]"><span>每步模拟</span><span className="font-mono">{simulations}</span></div>
                  <Slider value={[simulations]} min={20} max={300} step={20} onValueChange={([value]) => setSimulations(value)} />
                </div>
                <div>
                  <div className="mb-1 flex justify-between text-[10px] text-[#A79FC7]"><span>探索系数 c</span><span className="font-mono">{cPuct.toFixed(1)}</span></div>
                  <Slider value={[cPuct]} min={0.5} max={3} step={0.1} onValueChange={([value]) => setCPuct(value)} />
                </div>
              </div>
              {outcome !== null && (
                <div className="rounded-md border border-[#2A2445] bg-[#0D0B18] px-3 py-2 text-xs" style={{ color: outcome === 0 ? BLUE : outcome === 1 ? GOLD : PURPLE }}>
                  {outcome === 0 ? "平局" : outcome === 1 ? "× 获胜" : "○ 获胜"}
                </div>
              )}
              {/* 先验 / 访问分布紧凑并排，紧跟主棋盘下方 */}
              <div className="flex flex-wrap gap-4 border-t border-[#2A2445] pt-3">
                <div>
                  <div className="mb-1.5 flex items-center gap-1.5 text-[11px] text-[#A79FC7]"><BrainCircuit className="h-3.5 w-3.5 text-[#8B5CF6]" />网络先验 p(s,a)</div>
                  <BoardGrid board={board} values={pv.probs} compact />
                  <div className="mt-1.5 font-mono text-[10px] text-[#6B6390]">价值头 v(s) = {pv.value.toFixed(3)}</div>
                </div>
                <div>
                  <div className="mb-1.5 flex items-center gap-1.5 text-[11px] text-[#A79FC7]"><Search className="h-3.5 w-3.5 text-[#FFD21F]" />MCTS 访问分布 π(s,a)</div>
                  <BoardGrid board={board} values={visitDist} compact />
                  <div className="mt-1.5 font-mono text-[10px] text-[#6B6390]">训练目标：让 p 靠近 π，让 v 靠近终局 z</div>
                </div>
              </div>
            </div>

            <ActionStatsTable stats={stats} emptyHint="运行 MCTS 后，这里会显示每个候选动作的访问次数、价值、先验和 PUCT 分数。" />
          </div>
        </div>
      ) : tab === "train" ? (
        <div className="space-y-4 p-4">
          <div className="rounded-xl border border-[#2A2445] bg-[#0D0B18] p-3">
            <div className="flex flex-wrap items-center gap-2">
              <Button size="sm" className="bg-[#FFD21F] text-black hover:bg-[#FFE873]" disabled={iteration >= 100} onClick={() => setPlaying((value) => !value)}>
                {playing ? <Pause className="mr-1 h-4 w-4" /> : <Play className="mr-1 h-4 w-4" />}
                {playing ? "暂停" : "播放"}
              </Button>
              <Button size="sm" variant="outline" disabled={playing || iteration >= 100} onClick={() => advance(1)}>
                <StepForward className="mr-1 h-4 w-4" />单手落子
              </Button>
              <Button size="sm" variant="outline" onClick={runEvaluation}><Trophy className="mr-1 h-4 w-4" />立即评估</Button>
              <Button size="sm" variant="outline" onClick={resetTraining}><RotateCcw className="mr-1 h-4 w-4" />重置</Button>
              <span className="ml-auto font-mono text-[10px] text-[#6B6390]">ITERATION {iteration} / 100</span>
            </div>
            <div className="mt-3 grid grid-cols-2 gap-3 md:grid-cols-5">
              <Metric label="样本" value={String(samples)} accent={BLUE} />
              <Metric label="损失" value={loss.toFixed(3)} accent={PURPLE} />
              <Metric label="策略胜局" value={evaluating ? "评估中…" : `${latestEval.policyWin} / 12`} accent={GOLD} />
              <Metric label="策略负局" value={evaluating ? "…" : `${latestEval.policyLoss} / 12`} accent={MELON} />
              <Metric label="MCTS 胜局" value={evaluating ? "评估中…" : `${latestEval.mctsWin} / 6`} accent={BLUE} />
            </div>
            {evaluating && (
              <div className="mt-2 text-[11px] text-[#A79FC7]">正在用深度受限 minimax 对弈 18 局做评估，需要几秒钟——评估只在此刻发生，不影响训练。</div>
            )}
          </div>

          {/* 自我对弈实时面板：棋盘 + 每手根节点统计 + 训练曲线 */}
          <div className="grid grid-cols-1 gap-4 rounded-xl border border-[#2A2445] bg-[#0D0B18] p-3 xl:grid-cols-[auto_minmax(0,1fr)_minmax(0,1fr)]">
            <div className="space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-[#A79FC7]">
                <span>自我对弈实时棋盘（格内数字 = 该手 MCTS 访问数）</span>
                <span className="font-mono">
                  第 {liveGames} / 24 局 · 轮到 <b style={{ color: livePlayer === 1 ? GOLD : PURPLE }}>{livePlayer === 1 ? "× 先手" : "○ 后手"}</b>
                </span>
              </div>
              <BoardGrid board={liveBoard} visits={liveVisits} />
              <div>
                <div className="mb-1 flex justify-between text-[10px] text-[#A79FC7]"><span>动画速度（步/帧）</span><span className="font-mono">{movesPerTick}</span></div>
                <Slider value={[movesPerTick]} min={1} max={30} step={1} onValueChange={([value]) => { setMovesPerTick(value); movesPerTickRef.current = value; }} />
              </div>
            </div>
            <ActionStatsTable stats={liveStats} emptyHint="点击播放后，这里随每一手实时更新根节点统计（N / Q / P / PUCT）。" />
            <div className="space-y-4">
              <div>
                <div className="mb-2 text-xs text-[#A79FC7]">联合训练损失：策略交叉熵 + 价值 MSE</div>
                <div className="h-44 rounded-lg bg-[#07060E] p-2">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={history} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,92,246,0.12)" />
                      <XAxis dataKey="iteration" stroke="#6B6390" fontSize={10} />
                      <YAxis stroke="#6B6390" fontSize={10} width={42} domain={["auto", "auto"]} />
                      <Tooltip contentStyle={{ backgroundColor: "#0D0B18", border: "1px solid #2A2445", fontSize: 11 }} />
                      <Line type="monotone" dataKey="loss" name="联合损失" stroke={PURPLE} dot={false} strokeWidth={2} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div>
                <div className="mb-2 text-xs text-[#A79FC7]">对深度受限 solver 的胜局数：原始策略 vs MCTS</div>
                <div className="h-44 rounded-lg bg-[#07060E] p-2">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={history} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,92,246,0.12)" />
                      <XAxis dataKey="iteration" stroke="#6B6390" fontSize={10} />
                      <YAxis stroke="#6B6390" fontSize={10} width={42} domain={[0, 12]} />
                      <Tooltip contentStyle={{ backgroundColor: "#0D0B18", border: "1px solid #2A2445", fontSize: 11 }} />
                      <Line type="monotone" dataKey="policyWin" name="原始策略胜局" stroke={GOLD} dot={false} strokeWidth={2.4} isAnimationActive={false} />
                      <Line type="monotone" dataKey="mctsWin" name="MCTS 胜局" stroke={BLUE} dot={false} strokeWidth={1.8} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-2 text-[10px] leading-relaxed text-[#6B6390] md:grid-cols-2">
            <p>每次迭代包含 24 局自我对弈、每步 300 次 PUCT 模拟、10 轮策略价值训练。播放时上方棋盘实时展示当前对局，右表逐手刷新根节点统计；调大「动画速度」可加速跑完迭代。切回搜索页，可以直接比较训练后的网络先验与 MCTS 访问分布。</p>
            <p>评估与训练严格分离：评估器为深度受限 minimax，仅在评估时使用。目标不是证明小网络“无敌”，而是观察它如何从随机策略逐步获得与搜索一致的决策倾向。</p>
          </div>
        </div>
      ) : (
        <div className="space-y-4 p-4">
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-[auto_minmax(0,1fr)]">
            <div className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-[#A79FC7]">
                <span>
                  {playOutcome !== null
                    ? "本局结束"
                    : thinking
                      ? "智能体思考中…"
                      : playTurn === humanSide
                        ? "轮到你落子"
                        : ""}
                </span>
                <span className="font-mono">
                  你执 <b style={{ color: humanSide === 1 ? GOLD : PURPLE }}>{humanSide === 1 ? "× 先手" : "○ 后手"}</b>
                </span>
              </div>
              <BoardGrid board={playBoard} onCellClick={playOutcome === null && playTurn === humanSide && !thinking ? playMove : undefined} />
              {playOutcome !== null && (
                <div className="rounded-md border border-[#2A2445] bg-[#0D0B18] px-3 py-2 text-xs" style={{ color: playOutcome === 0 ? BLUE : playOutcome === humanSide ? GOLD : MELON }}>
                  {playOutcome === 0 ? "平局" : playOutcome === humanSide ? "你赢了！" : "智能体获胜"}
                </div>
              )}
            </div>

            <div className="space-y-3">
              <div className="rounded-xl border border-[#2A2445] bg-[#0D0B18] p-3 space-y-3">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-[11px] text-[#A79FC7]">执子：</span>
                  <Button size="sm" variant={humanSide === 1 ? "default" : "outline"} onClick={() => resetPlay(1)}>× 先手</Button>
                  <Button size="sm" variant={humanSide === -1 ? "default" : "outline"} onClick={() => resetPlay(-1)}>○ 后手</Button>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-[11px] text-[#A79FC7]">对手：</span>
                  <Button size="sm" variant={agentMode === "mcts" ? "default" : "outline"} onClick={() => setAgentMode("mcts")}>MCTS + 网络（强）</Button>
                  <Button size="sm" variant={agentMode === "net" ? "default" : "outline"} onClick={() => setAgentMode("net")}>纯网络 argmax（快）</Button>
                </div>
                <div className="flex items-center gap-2">
                  <Button size="sm" className="bg-[#FFD21F] text-black hover:bg-[#FFE873]" onClick={() => resetPlay(humanSide)}>
                    <RotateCcw className="mr-1 h-4 w-4" />再来一局
                  </Button>
                </div>
                <div className="grid grid-cols-3 gap-2 pt-1 text-center">
                  <Metric label="你的胜局" value={String(score.win)} accent={GOLD} />
                  <Metric label="平局" value={String(score.draw)} accent={BLUE} />
                  <Metric label="你的负局" value={String(score.loss)} accent={MELON} />
                </div>
              </div>
              <div className="rounded-lg border border-[#2A2445] bg-[#0D0B18] p-3 text-[11px] leading-relaxed text-[#A79FC7]">
                对手就是当前内存中的同一个网络：先去「② 自我对弈训练」页跑几轮迭代再回来，会明显感觉它变强了——这正是「搜索作为策略改进算子」的直接体验。未训练时它只是随机初始化的弱对手。「MCTS + 网络」档还会叠加 100 次 PUCT 模拟，比纯网络 argmax 更难对付。
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
