import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, CartesianGrid } from "recharts";
import { Play, Pause, StepForward, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { GridWorldModel } from "@/dp/GridWorldModel";
import { makeRng, sampleEpisode, FirstVisitMC, TDZero, TDLambda, evaluatePolicyExact, type Trajectory } from "@/dp/MCTD";
import ValueHeatmap from "@/components/ValueHeatmap";
import TeX from "@/components/TeX";

const WALLS = [6, 7, 11];
const OBSERVE_STATE = 0; // 起点：学习曲线的观测状态

interface HistPoint {
  ep: number;
  mc: number;
  td: number;
  tdl: number;
}

interface View {
  mc: number[];
  td: number[];
  tdl: number[];
  ep: number;
  history: HistPoint[];
  traj: Trajectory | null;
  tdSteps: number;      // 当前 episode 中 TD 已更新的步数
  mcPending: boolean;   // MC 是否在等待终点
  mcUpdated: number;    // 最近一次 episode 结束 MC 回溯更新的状态数
}

export default function MCTDDemo() {
  const [gamma, setGamma] = useState(0.99);
  const [alpha, setAlpha] = useState(0.1);
  const [lam, setLam] = useState(0.0);
  const [speed, setSpeed] = useState(200); // ms / episode
  const [playing, setPlaying] = useState(false);
  const [view, setView] = useState<View>(() => initView(0.99, 0.1, 0.0));

  const mcRef = useRef<FirstVisitMC | null>(null);
  const tdRef = useRef<TDZero | null>(null);
  const tdlRef = useRef<TDLambda | null>(null);
  const rngRef = useRef<() => number>(() => 0);

  const model = useMemo(() => new GridWorldModel(5, WALLS), []);
  const truth = useMemo(() => evaluatePolicyExact(model, gamma), [model, gamma]);
  const vmax = useMemo(() => Math.max(...truth), [truth]);

  function initView(_g: number, _a: number, _l: number): View {
    const m = new GridWorldModel(5, WALLS);
    return {
      mc: new Array(m.stateCount).fill(0),
      td: new Array(m.stateCount).fill(0),
      tdl: new Array(m.stateCount).fill(0),
      ep: 0,
      history: [{ ep: 0, mc: 0, td: 0, tdl: 0 }],
      traj: null,
      tdSteps: 0,
      mcPending: false,
      mcUpdated: 0,
    };
  }

  const reset = useCallback((g: number, a: number, l: number) => {
    setPlaying(false);
    mcRef.current = new FirstVisitMC(new GridWorldModel(5, WALLS));
    tdRef.current = new TDZero(new GridWorldModel(5, WALLS), a);
    tdlRef.current = new TDLambda(new GridWorldModel(5, WALLS), a, l);
    rngRef.current = makeRng(20261002);
    setView(initView(g, a, l));
  }, []);

  // 首次挂载初始化实例
  useEffect(() => {
    reset(gamma, alpha, lam);
  }, []);

  const runEpisode = useCallback(() => {
    const mc = mcRef.current, td = tdRef.current, tdl = tdlRef.current, rng = rngRef.current;
    if (!mc || !td || !tdl || !rng) return;
    const g = gamma;
    const traj = sampleEpisode(new GridWorldModel(5, WALLS), rng);
    // 同一条轨迹喂给三个估计器
    const mcUpdated = mc.observeEpisode(traj, g);
    let steps = 0;
    for (let t = 0; t < traj.states.length - 1; t++) {
      td.observeStep(traj.states[t], traj.rewards[t], traj.states[t + 1], traj.states[t + 1] === 24, g);
      tdl.observeStep(traj.states[t], traj.rewards[t], traj.states[t + 1], traj.states[t + 1] === 24, g);
      steps++;
    }
    setView(prev => {
      const ep = prev.ep + 1;
      const point: HistPoint = { ep, mc: mc.v[OBSERVE_STATE], td: td.v[OBSERVE_STATE], tdl: tdl.v[OBSERVE_STATE] };
      const history = [...prev.history, point];
      if (history.length > 600) history.shift();
      return {
        mc: [...mc.v], td: [...td.v], tdl: [...tdl.v],
        ep, history, traj, tdSteps: steps, mcPending: false, mcUpdated: mcUpdated.length,
      };
    });
  }, [gamma]);

  useEffect(() => {
    if (!playing) return;
    const t = setInterval(runEpisode, speed);
    return () => clearInterval(t);
  }, [playing, speed, runEpisode]);

  const truth0 = truth[OBSERVE_STATE];

  return (
    <div className="rounded-xl border border-border bg-[#0A0A0F] shadow-glow-game overflow-hidden">
      <div className="px-4 py-3 border-b border-border/60 flex flex-wrap items-center justify-between gap-3">
        <div className="text-sm text-muted-foreground">
          <TeX math="V(S_t)\leftarrow V(S_t)+\alpha\big[\underbrace{R+\gamma V(S_{t+1})-V(S_t)}_{\text{TD error}}\big]" />
        </div>
        <span className="text-xs font-mono text-text-muted">同一条轨迹，喂给三个估计器</span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 p-4">
        {/* 双地图对比 */}
        <div className="space-y-3">
          <div>
            <div className="text-xs text-accent-watermelon font-semibold mb-1">MC 眼中的地图（等结局才更新）</div>
            <div className="flex justify-center bg-black/30 rounded-lg p-3">
              <ValueHeatmap size={5} values={view.mc} walls={WALLS} goal={24} vmin={0} vmax={vmax} showValues={false} />
            </div>
          </div>
          <div>
            <div className="text-xs text-accent-yellow font-semibold mb-1">
              {lam === 0 ? "TD(0) 眼中的地图（走一步更新一步）" : `TD(λ=${lam.toFixed(1)}) 眼中的地图`}
            </div>
            <div className="flex justify-center bg-black/30 rounded-lg p-3">
              <ValueHeatmap size={5} values={lam === 0 ? view.td : view.tdl} walls={WALLS} goal={24} vmin={0} vmax={vmax} showValues={false} />
            </div>
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
            <Button size="sm" variant="outline" onClick={() => reset(gamma, alpha, lam)}>
              <RotateCcw className="w-4 h-4 mr-1" /> 重置
            </Button>
            <span className="text-xs font-mono text-text-muted ml-auto">episode {view.ep}</span>
          </div>

          <div className="grid grid-cols-2 gap-3 text-center text-sm">
            <div className="rounded-md bg-black/30 border border-border/50 py-2">
              <div className="text-muted-foreground text-xs">TD 已在线更新</div>
              <div className="font-mono text-lg text-accent-yellow">{view.history.length > 1 ? view.tdSteps : "—"} 步 / 条轨迹</div>
            </div>
            <div className="rounded-md bg-black/30 border border-border/50 py-2">
              <div className="text-muted-foreground text-xs">MC 等待终点后回溯</div>
              <div className="font-mono text-lg text-accent-watermelon">
                {view.mcUpdated > 0 ? `+${view.mcUpdated} 个状态` : "等待中…"}
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-xs text-muted-foreground mb-1"><span>折扣 γ = {gamma.toFixed(2)}</span></div>
              <Slider value={[gamma]} min={0.5} max={0.99} step={0.01} onValueChange={([g]) => { setGamma(g); reset(g, alpha, lam); }} />
            </div>
            <div>
              <div className="flex justify-between text-xs text-muted-foreground mb-1"><span>步长 α = {alpha.toFixed(2)}</span></div>
              <Slider value={[alpha]} min={0.02} max={0.5} step={0.01} onValueChange={([a]) => { setAlpha(a); reset(gamma, a, lam); }} />
            </div>
            <div>
              <div className="flex justify-between text-xs text-muted-foreground mb-1"><span>λ = {lam.toFixed(1)}（0=TD(0)，1→MC）</span></div>
              <Slider value={[lam]} min={0} max={1} step={0.1} onValueChange={([l]) => { setLam(l); reset(gamma, alpha, l); }} />
            </div>
            <div>
              <div className="flex justify-between text-xs text-muted-foreground mb-1"><span>速度 {speed} ms/条</span></div>
              <Slider value={[speed]} min={40} max={800} step={20} onValueChange={([s]) => setSpeed(s)} />
            </div>
          </div>

          <div className="h-44">
            <div className="text-xs text-muted-foreground mb-1">起点 V(s₀) 的估计值随轨迹数变化（白色虚线 = DP 真值 {truth0.toFixed(2)}）</div>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={view.history} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,92,246,0.15)" />
                <XAxis dataKey="ep" stroke="#6B7280" fontSize={10} />
                <YAxis domain={["auto", "auto"]} stroke="#6B7280" fontSize={10} width={44}
                  tickFormatter={v => v.toFixed(1)} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#0A0A0F", border: "1px solid rgba(139,92,246,0.3)", fontSize: 12 }}
                  formatter={(v: number, name: string) => [v.toFixed(3), name]}
                  labelFormatter={l => `episode ${l}`}
                />
                <ReferenceLine y={truth0} stroke="#fff" strokeDasharray="4 4" strokeOpacity={0.5} />
                <Line type="stepAfter" dataKey="mc" name="MC（无偏·高方差）" stroke="#FF6B6B" dot={false} strokeWidth={2} isAnimationActive={false} />
                {lam === 0 ? (
                  <Line type="stepAfter" dataKey="td" name="TD(0)（低方差·有偏）" stroke="#FACC15" dot={false} strokeWidth={2} isAnimationActive={false} />
                ) : (
                  <Line type="stepAfter" dataKey="tdl" name={`TD(λ=${lam.toFixed(1)})`} stroke="#FACC15" dot={false} strokeWidth={2} isAnimationActive={false} />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
