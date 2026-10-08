import { useCallback, useEffect, useRef, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { Play, Pause, StepForward, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { makeRng } from "@/dp/MCTD";
import {
  CW_ROWS, CW_COLS, CW_START, CW_GOAL, CW_CLIFF,
  TabularQLearning, TabularSARSA, runEpisodeQLearning, runEpisodeSARSA,
} from "@/dp/CliffWalk";
import TeX from "@/components/TeX";

const ARROWS = ["▲", "▼", "◀", "▶"];
const MA = 50; // 滑动平均窗口

interface HistPoint {
  ep: number;
  ql: number | null;
  sa: number | null;
}

interface View {
  qlPi: number[];
  saPi: number[];
  ep: number;
  history: HistPoint[];
  lastQl: number;
  lastSa: number;
}

function initView(): View {
  return {
    qlPi: new Array(CW_ROWS * CW_COLS).fill(-1),
    saPi: new Array(CW_ROWS * CW_COLS).fill(-1),
    ep: 0,
    history: [{ ep: 0, ql: null, sa: null }],
    lastQl: 0,
    lastSa: 0,
  };
}

/** 内联渲染悬崖地图：红色悬崖行 + 贪心策略箭头；pathClass 控制箭头配色 */
function CliffMap({ pi, accent }: { pi: number[]; accent: string }) {
  return (
    <div className="grid gap-0.5 select-none" style={{ gridTemplateColumns: `repeat(${CW_COLS}, minmax(0,1fr))` }}>
      {Array.from({ length: CW_ROWS * CW_COLS }, (_, s) => {
        const isCliff = CW_CLIFF.has(s);
        const isGoal = s === CW_GOAL;
        const isStart = s === CW_START;
        return (
          <div
            key={s}
            className="aspect-square rounded-[2px] flex items-center justify-center text-[8px] sm:text-[10px]"
            style={{
              backgroundColor: isCliff ? "#E0304A" : "#12121A",
              border: "1px solid rgba(139,92,246,0.15)",
              color: accent,
            }}
          >
            {isGoal ? <span style={{ color: "#F0B822" }}>★</span>
              : isStart ? <span style={{ color: "#F0B822" }}>S</span>
              : isCliff ? null
              : pi[s] >= 0 ? ARROWS[pi[s]] : null}
          </div>
        );
      })}
    </div>
  );
}

export default function CliffWalkDemo() {
  const [eps, setEps] = useState(0.1);
  const [alpha, setAlpha] = useState(0.5);
  const [speed, setSpeed] = useState(60); // ms / episode
  const [playing, setPlaying] = useState(false);
  const [view, setView] = useState<View>(initView);

  const qlRef = useRef<TabularQLearning | null>(null);
  const saRef = useRef<TabularSARSA | null>(null);
  const rngQRef = useRef<() => number>(() => 0);
  const rngSRef = useRef<() => number>(() => 0);
  const rewardsRef = useRef<{ ql: number[]; sa: number[] }>({ ql: [], sa: [] });

  const reset = useCallback((a: number) => {
    setPlaying(false);
    qlRef.current = new TabularQLearning(a);
    saRef.current = new TabularSARSA(a);
    rngQRef.current = makeRng(777);
    rngSRef.current = makeRng(777);
    rewardsRef.current = { ql: [], sa: [] };
    setView(initView());
  }, []);

  useEffect(() => { reset(alpha); }, []);

  const movingAvg = (arr: number[], k: number) => {
    const start = Math.max(0, arr.length - MA);
    const slice = arr.slice(start, k);
    if (slice.length === 0) return null;
    return slice.reduce((x, y) => x + y, 0) / slice.length;
  };

  const runEpisode = useCallback(() => {
    const ql = qlRef.current, sa = saRef.current;
    if (!ql || !sa) return;
    const rQ = runEpisodeQLearning(ql, eps, 1.0, rngQRef.current);
    const rS = runEpisodeSARSA(sa, eps, 1.0, rngSRef.current);
    rewardsRef.current.ql.push(rQ);
    rewardsRef.current.sa.push(rS);
    const rw = rewardsRef.current;
    setView(prev => {
      const ep = prev.ep + 1;
      const point: HistPoint = {
        ep,
        ql: rw.ql.length >= MA / 2 ? movingAvg(rw.ql, rw.ql.length) : null,
        sa: rw.sa.length >= MA / 2 ? movingAvg(rw.sa, rw.sa.length) : null,
      };
      const history = [...prev.history, point];
      if (history.length > 800) history.shift();
      return {
        qlPi: ql.greedyPolicy(), saPi: sa.greedyPolicy(),
        ep, history, lastQl: rQ, lastSa: rS,
      };
    });
  }, [eps]);

  useEffect(() => {
    if (!playing) return;
    const t = setInterval(runEpisode, speed);
    return () => clearInterval(t);
  }, [playing, speed, runEpisode]);

  return (
    <div className="rounded-xl border border-border bg-[#0A0A0F] shadow-glow-game overflow-hidden">
      <div className="px-4 py-3 border-b border-border/60 flex flex-wrap items-center justify-between gap-3">
        <div className="text-sm text-muted-foreground">
          <TeX math="Q(S_t,A_t)\leftarrow Q+\alpha\big[R+\gamma\max_{a'}Q(S_{t+1},a')-Q\big]" />
        </div>
        <span className="text-xs font-mono text-text-muted">Sutton &amp; Barto 例 6.6 · γ=1</span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 p-4">
        {/* 双贪心路径地图 */}
        <div className="space-y-3">
          <div>
            <div className="text-xs text-accent-yellow font-semibold mb-1">Q-learning 的贪心路径（贴崖·最优·危险）</div>
            <div className="bg-black/30 rounded-lg p-3">
              <CliffMap pi={view.qlPi} accent="#F0B822" />
            </div>
          </div>
          <div>
            <div className="text-xs text-accent-purple font-semibold mb-1">SARSA 的贪心路径（绕远·安全·次优）</div>
            <div className="bg-black/30 rounded-lg p-3">
              <CliffMap pi={view.saPi} accent="#A78BFA" />
            </div>
          </div>
          <div className="text-[11px] text-text-muted leading-relaxed">
            起点 S 在左下，金色 ★ 在右下，<span className="text-accent-watermelon">红色一排是悬崖</span>（踩上 = −100 并回到 S）。
            观察两张地图箭头收敛后的形状差异：一条贴着悬崖走，一条敬而远之。
          </div>
        </div>

        {/* 控制 + 曲线 */}
        <div className="space-y-4">
          <div className="flex items-center gap-2 flex-wrap">
            <Button size="sm" onClick={() => setPlaying(p => !p)}>
              {playing ? <Pause className="w-4 h-4 mr-1" /> : <Play className="w-4 h-4 mr-1" />}
              {playing ? "暂停" : "播放"}
            </Button>
            <Button size="sm" variant="outline" onClick={runEpisode}>
              <StepForward className="w-4 h-4 mr-1" /> 单条轨迹
            </Button>
            <Button size="sm" variant="outline" onClick={() => reset(alpha)}>
              <RotateCcw className="w-4 h-4 mr-1" /> 重置
            </Button>
            <span className="text-xs font-mono text-text-muted ml-auto">episode {view.ep}</span>
          </div>

          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-xs text-muted-foreground mb-1"><span>探索率 ε = {eps.toFixed(2)}</span></div>
              <Slider value={[eps]} min={0.01} max={0.3} step={0.01} onValueChange={([e]) => setEps(e)} />
            </div>
            <div>
              <div className="flex justify-between text-xs text-muted-foreground mb-1"><span>步长 α = {alpha.toFixed(2)}</span></div>
              <Slider value={[alpha]} min={0.1} max={0.9} step={0.05} onValueChange={([a]) => { setAlpha(a); reset(a); }} />
            </div>
            <div>
              <div className="flex justify-between text-xs text-muted-foreground mb-1"><span>速度 {speed} ms/条</span></div>
              <Slider value={[speed]} min={20} max={300} step={10} onValueChange={([s]) => setSpeed(s)} />
            </div>
          </div>

          <div className="h-48">
            <div className="text-xs text-muted-foreground mb-1">在线性能：最近 {MA} 条轨迹的平均累积奖励（越高越好）</div>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={view.history} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,92,246,0.15)" />
                <XAxis dataKey="ep" stroke="#6B7280" fontSize={10} />
                <YAxis domain={[-120, -10]} stroke="#6B7280" fontSize={10} width={44} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#0A0A0F", border: "1px solid rgba(139,92,246,0.3)", fontSize: 12 }}
                  formatter={(v: number, name: string) => [v?.toFixed(1), name]}
                  labelFormatter={l => `episode ${l}`}
                />
                <Line type="monotone" dataKey="ql" name="Q-learning（在线）" stroke="#F0B822" dot={false} strokeWidth={2} isAnimationActive={false} connectNulls />
                <Line type="monotone" dataKey="sa" name="SARSA（在线）" stroke="#8B5CF6" dot={false} strokeWidth={2} isAnimationActive={false} connectNulls />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="grid grid-cols-2 gap-3 text-center text-sm">
            <div className="rounded-md bg-black/30 border border-border/50 py-2">
              <div className="text-muted-foreground text-xs">上一条轨迹奖励</div>
              <div className="font-mono text-lg text-accent-yellow">{view.lastQl.toFixed(0)}</div>
            </div>
            <div className="rounded-md bg-black/30 border border-border/50 py-2">
              <div className="text-muted-foreground text-xs">上一条轨迹奖励</div>
              <div className="font-mono text-lg text-accent-purple">{view.lastSa.toFixed(0)}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
