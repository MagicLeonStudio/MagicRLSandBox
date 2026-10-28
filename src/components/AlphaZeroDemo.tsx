import { useCallback, useEffect, useRef, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { Play, Pause, StepForward, RotateCcw, Search, BrainCircuit, Trophy } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import TeX from "@/components/TeX";
import { PolicyValueNet, MiniMaxSolver, runMCTS, selfPlayIteration, evaluateAgainstSolver, winner, CELL_COUNT } from "@/dp/TinyAlphaZero";
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
}: {
  board: Board;
  onCellClick?: (index: number) => void;
  values?: number[];
  visits?: number[];
}) {
  return (
    <div className="grid w-fit grid-cols-5 gap-1.5 rounded-lg border border-[#2A2445] bg-[#07060E] p-2">
      {Array.from({ length: CELL_COUNT }, (_, index) => {
        const stone = board[index];
        const value = values?.[index];
        const visit = visits?.[index];
        return (
          <button
            key={index}
            disabled={stone !== 0 || !onCellClick}
            onClick={() => onCellClick?.(index)}
            className="relative flex h-11 w-11 items-center justify-center rounded-md border text-sm font-bold transition-colors disabled:cursor-default md:h-14 md:w-14 md:text-base"
            style={{
              borderColor: stone === 1 ? GOLD : stone === -1 ? PURPLE : "#2A2445",
              backgroundColor: value !== undefined ? `rgba(255,210,31,${Math.min(0.55, value * 0.55)})` : stone === 0 ? "#0D0B18" : "#141126",
            }}
          >
            <span style={{ color: stone === 1 ? GOLD : stone === -1 ? PURPLE : "#6B6390" }}>
              {stone === 1 ? "×" : stone === -1 ? "○" : ""}
            </span>
            {value !== undefined && <span className="absolute bottom-0.5 right-1 font-mono text-[9px] text-[#F1EEFB]">{(value * 100).toFixed(0)}%</span>}
            {visit !== undefined && <span className="absolute bottom-0.5 right-1 font-mono text-[9px] text-[#F1EEFB]">{visit}</span>}
          </button>
        );
      })}
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

export default function AlphaZeroDemo() {
  const [tab, setTab] = useState<"search" | "train">("search");
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
  const [latestEval, setLatestEval] = useState<{ policyWin: number; policyLoss: number; mctsWin: number; mctsLoss: number }>({ policyWin: 0, policyLoss: 0, mctsWin: 0, mctsLoss: 0 });
  const iterationRef = useRef(0);

  const runEvaluation = useCallback(() => {
    solver.maxDepth = 6;
    const policy = evaluateAgainstSolver(net, solver, 12, 0, 42);
    const mcts = evaluateAgainstSolver(net, solver, 6, 100, 42);
    const point = { policyWin: policy.win, policyLoss: policy.loss, mctsWin: mcts.win, mctsLoss: mcts.loss };
    setLatestEval(point);
    return point;
  }, [net, solver]);

  useEffect(() => { runEvaluation(); }, []);

  const runIteration = useCallback(() => {
    if (iterationRef.current >= 100) return;
    const next = iterationRef.current + 1;
    const info = selfPlayIteration(net, { games: 24, simulations: 300, epochs: 10, seed: next });
    iterationRef.current = next;
    setIteration(next);
    setSamples(info.samples);
    setLoss(info.loss);
    let evaluation = latestEval;
    if (next % 10 === 0 || next === 100) evaluation = runEvaluation();
    const point: EvalHistory = {
      iteration: next,
      samples: info.samples,
      loss: info.loss,
      policyWin: evaluation.policyWin,
      policyLoss: evaluation.policyLoss,
      mctsWin: evaluation.mctsWin,
      mctsLoss: evaluation.mctsLoss,
    };
    setHistory((prev) => [...prev, point]);
    if (next >= 100) setPlaying(false);
  }, [net, latestEval, runEvaluation]);

  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(runIteration, 250);
    return () => clearInterval(timer);
  }, [playing, runIteration]);

  const resetTraining = () => {
    setPlaying(false);
    netRef.current = new PolicyValueNet(7);
    iterationRef.current = 0;
    setIteration(0);
    setSamples(0);
    setLoss(0);
    setHistory([]);
    runEvaluation();
  };

  const totalVisits = stats.reduce((sum, stat) => sum + stat.visits, 0);
  const visitDist = Array.from({ length: CELL_COUNT }, (_, action) => {
    const stat = stats.find((item) => item.action === action);
    return totalVisits > 0 && stat ? stat.visits / totalVisits : 0;
  });
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
          {(["search", "train"] as const).map((item) => (
            <button
              key={item}
              onClick={() => setTab(item)}
              className="flex-1 rounded-md px-3 py-2"
              style={{ backgroundColor: tab === item ? "#1C1832" : "transparent", color: tab === item ? GOLD : "#A79FC7" }}
            >
              {item === "search" ? "① PUCT 搜索与策略蒸馏" : "② 自我对弈训练与强度评估"}
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
            </div>

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
              {stats.length === 0 && <div className="py-6 text-center text-xs text-[#6B6390]">运行 MCTS 后，这里会显示每个候选动作的访问次数、价值、先验和 PUCT 分数。</div>}
            </div>
          </div>

          <div className="grid grid-cols-1 items-start gap-4 border-t border-[#2A2445] pt-4 md:grid-cols-2">
            <div>
              <div className="mb-2 flex items-center gap-2 text-xs text-[#A79FC7]"><BrainCircuit className="h-4 w-4 text-[#8B5CF6]" />网络先验 p(s,a)</div>
              <BoardGrid board={board} values={pv.probs} />
              <div className="mt-2 font-mono text-[10px] text-[#6B6390]">价值头 v(s) = {pv.value.toFixed(3)}</div>
            </div>
            <div>
              <div className="mb-2 flex items-center gap-2 text-xs text-[#A79FC7]"><Search className="h-4 w-4 text-[#FFD21F]" />MCTS 访问分布 π(s,a)</div>
              <BoardGrid board={board} values={visitDist} />
              <div className="mt-2 font-mono text-[10px] text-[#6B6390]">训练目标：让 p 靠近 π，让 v 靠近终局 z</div>
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-4 p-4">
          <div className="rounded-xl border border-[#2A2445] bg-[#0D0B18] p-3">
            <div className="flex flex-wrap items-center gap-2">
              <Button size="sm" className="bg-[#FFD21F] text-black hover:bg-[#FFE873]" disabled={iteration >= 100} onClick={() => setPlaying((value) => !value)}>
                {playing ? <Pause className="mr-1 h-4 w-4" /> : <Play className="mr-1 h-4 w-4" />}
                {playing ? "暂停" : "播放"}
              </Button>
              <Button size="sm" variant="outline" disabled={playing || iteration >= 100} onClick={runIteration}>
                <StepForward className="mr-1 h-4 w-4" />单次迭代
              </Button>
              <Button size="sm" variant="outline" onClick={runEvaluation}><Trophy className="mr-1 h-4 w-4" />立即评估</Button>
              <Button size="sm" variant="outline" onClick={resetTraining}><RotateCcw className="mr-1 h-4 w-4" />重置</Button>
              <span className="ml-auto font-mono text-[10px] text-[#6B6390]">ITERATION {iteration} / 100</span>
            </div>
            <div className="mt-3 grid grid-cols-2 gap-3 md:grid-cols-5">
              <Metric label="样本" value={String(samples)} accent={BLUE} />
              <Metric label="损失" value={loss.toFixed(3)} accent={PURPLE} />
              <Metric label="策略胜局" value={`${latestEval.policyWin} / 12`} accent={GOLD} />
              <Metric label="策略负局" value={`${latestEval.policyLoss} / 12`} accent={MELON} />
              <Metric label="MCTS 胜局" value={`${latestEval.mctsWin} / 6`} accent={BLUE} />
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <div>
              <div className="mb-2 text-xs text-[#A79FC7]">联合训练损失：策略交叉熵 + 价值 MSE</div>
              <div className="h-44 rounded-lg bg-[#0D0B18] p-2">
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
              <div className="h-44 rounded-lg bg-[#0D0B18] p-2">
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

          <div className="grid grid-cols-1 gap-2 text-[10px] leading-relaxed text-[#6B6390] md:grid-cols-2">
            <p>每次迭代包含 24 局自我对弈、每步 300 次 PUCT 模拟、10 轮策略价值训练。切回搜索页，可以直接比较训练后的网络先验与 MCTS 访问分布。</p>
            <p>评估与训练严格分离：评估器为深度受限 minimax，仅在评估时使用。目标不是证明小网络“无敌”，而是观察它如何从随机策略逐步获得与搜索一致的决策倾向。</p>
          </div>
        </div>
      )}
    </div>
  );
}
