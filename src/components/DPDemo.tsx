import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Play, Pause, StepForward, RotateCcw } from "lucide-react";
import { GridWorldModel } from "@/dp/GridWorldModel";
import { ValueIteration } from "@/dp/ValueIteration";
import { PolicyIteration } from "@/dp/PolicyIteration";
import ValueHeatmap from "@/components/ValueHeatmap";
import TeX from "@/components/TeX";

type Mode = "vi" | "pi";
interface View {
  v: number[];
  sweep: number;
  delta: number | null;
  done: boolean;
  policy: number[];
  evalDelta: number | null;
}

const WALL_LAYOUT: number[] = [6, 7, 11]; // 左下障碍块（不影响最短路，但塑造 V 值地形）

export default function DPDemo() {
  const viRef = useRef<ValueIteration | null>(null);
  const piRef = useRef<PolicyIteration | null>(null);

  const [mode, setMode] = useState<Mode>("vi");
  const [gamma, setGamma] = useState(0.99);
  const [useWalls, setUseWalls] = useState(true);
  const [speed, setSpeed] = useState(400); // ms / sweep
  const [playing, setPlaying] = useState(false);
  const [view, setView] = useState<View>(() => initView("vi", 0.99, true));

  function buildModel(walls: boolean) {
    return new GridWorldModel(5, walls ? WALL_LAYOUT : []);
  }

  function initView(m: Mode, g: number, walls: boolean): View {
    const model = buildModel(walls);
    if (m === "vi") {
      const vi = new ValueIteration(model, g, 1e-6);
      viRef.current = vi;
      piRef.current = null;
      return { v: [...vi.v], sweep: 0, delta: null, done: false, policy: vi.greedyPolicy(), evalDelta: null };
    }
    const pi = new PolicyIteration(model, g, 1e-4);
    piRef.current = pi;
    viRef.current = null;
    return { v: [...pi.v], sweep: 0, delta: null, done: false, policy: [...pi.policy], evalDelta: null };
  }

  const reset = useCallback((m: Mode, g: number, walls: boolean) => {
    setPlaying(false);
    setView(initView(m, g, walls));
  }, []);

  const stepOnce = useCallback(() => {
    setView(prev => {
      if (prev.done) return prev;
      if (mode === "vi" && viRef.current) {
        const r = viRef.current.sweep();
        return { v: r.v, sweep: r.sweep, delta: r.delta, done: r.done, policy: viRef.current.greedyPolicy(), evalDelta: null };
      }
      if (mode === "pi" && piRef.current) {
        const r = piRef.current.step();
        return { v: r.v, sweep: r.iteration, delta: r.evalDelta, done: r.done, policy: r.policy, evalDelta: r.evalDelta };
      }
      return prev;
    });
  }, [mode]);

  useEffect(() => {
    if (!playing) return;
    if (view.done) { setPlaying(false); return; }
    const t = setInterval(stepOnce, speed);
    return () => clearInterval(t);
  }, [playing, speed, stepOnce, view.done]);

  // 收敛后：沿贪心策略模拟路径用于高亮
  const path = useMemo(() => {
    if (!view.done) return undefined;
    const model = buildModel(useWalls);
    const p: number[] = [];
    let s = 0;
    for (let i = 0; i < 50 && !model.isTerminal(s); i++) {
      p.push(s);
      const a = view.policy[s];
      if (a < 0) break;
      s = model.nextState(s, a);
    }
    p.push(s);
    return p;
  }, [view.done, view.policy, useWalls]);

  const chartData = useMemo(() => {
    const src = mode === "vi" ? viRef.current?.deltas ?? [] : piRef.current?.history.map(h => h.evalDelta) ?? [];
    return src.map((d, i) => ({ sweep: i + 1, delta: Math.max(d, 1e-12) }));
  }, [view, mode]);

  return (
    <div className="rounded-xl border border-border bg-[#0A0A0F] shadow-glow-game overflow-hidden">
      <div className="px-4 py-3 border-b border-border/60 flex flex-wrap items-center justify-between gap-3">
        <Tabs value={mode} onValueChange={v => { setMode(v as Mode); reset(v as Mode, gamma, useWalls); }}>
          <TabsList>
            <TabsTrigger value="vi">价值迭代 VI</TabsTrigger>
            <TabsTrigger value="pi">策略迭代 PI</TabsTrigger>
          </TabsList>
        </Tabs>
        <div className="text-sm text-muted-foreground">
          <TeX math="V_{k+1}(s)=\max_a\sum_{s'}P\big[r+\gamma V_k(s')\big]" />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 p-4">
        <div className="flex items-center justify-center bg-black/30 rounded-lg p-4">
          <ValueHeatmap
            size={5}
            values={view.v}
            policy={view.policy}
            walls={useWalls ? WALL_LAYOUT : []}
            goal={24}
            highlightPath={path}
          />
        </div>

        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <Button size="sm" onClick={() => setPlaying(p => !p)} disabled={view.done && !playing}>
              {playing ? <Pause className="w-4 h-4 mr-1" /> : <Play className="w-4 h-4 mr-1" />}
              {playing ? "暂停" : "播放"}
            </Button>
            <Button size="sm" variant="outline" onClick={stepOnce} disabled={view.done}>
              <StepForward className="w-4 h-4 mr-1" /> 单步
            </Button>
            <Button size="sm" variant="outline" onClick={() => reset(mode, gamma, useWalls)}>
              <RotateCcw className="w-4 h-4 mr-1" /> 重置
            </Button>
          </div>

          <div className="grid grid-cols-3 gap-2 text-center text-sm">
            <div className="rounded-md bg-black/30 border border-border/50 py-2">
              <div className="text-muted-foreground text-xs">{mode === "vi" ? "Sweep 轮数" : "迭代轮数"}</div>
              <div className="font-mono text-lg text-accent-yellow">{view.sweep}</div>
            </div>
            <div className="rounded-md bg-black/30 border border-border/50 py-2">
              <div className="text-muted-foreground text-xs">{mode === "vi" ? "Δ（残差）" : "评估 Δ"}</div>
              <div className="font-mono text-lg text-accent-purple">
                {view.delta === null ? "—" : view.delta < 1e-6 ? "≈0" : view.delta.toFixed(4)}
              </div>
            </div>
            <div className="rounded-md bg-black/30 border border-border/50 py-2">
              <div className="text-muted-foreground text-xs">状态</div>
              <div className={`font-mono text-lg ${view.done ? "text-accent-yellow" : "text-accent-purple"}`}>
                {view.done ? "已收敛" : "迭代中"}
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-xs text-muted-foreground mb-1">
                <span>折扣因子 γ = {gamma.toFixed(2)}</span>
              </div>
              <Slider value={[gamma]} min={0.5} max={0.99} step={0.01} onValueChange={([g]) => { setGamma(g); reset(mode, g, useWalls); }} />
            </div>
            <div>
              <div className="flex justify-between text-xs text-muted-foreground mb-1">
                <span>播放速度 {speed} ms/轮</span>
              </div>
              <Slider value={[speed]} min={60} max={1000} step={20} onValueChange={([s]) => setSpeed(s)} />
            </div>
            <div className="flex items-center gap-2 text-sm">
              <Button size="sm" variant={useWalls ? "default" : "outline"} onClick={() => { setUseWalls(w => !w); reset(mode, gamma, !useWalls); }}>
                障碍布局 {useWalls ? "开" : "关"}
              </Button>
            </div>
          </div>

          <div className="h-36">
            <div className="text-xs text-muted-foreground mb-1">Bellman 残差 Δ 随轮数（每次扫描后 max|V′−V|）</div>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,92,246,0.15)" />
                <XAxis dataKey="sweep" stroke="#6B7280" fontSize={10} />
                <YAxis scale="log" domain={[1e-12, "auto"]} stroke="#6B7280" fontSize={10} tickFormatter={v => (v >= 1 ? v.toFixed(0) : v.toExponential(0))} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#0A0A0F", border: "1px solid rgba(139,92,246,0.3)", fontSize: 12 }}
                  formatter={(v: number) => [v.toExponential(2), "Δ"]}
                />
                <Line type="monotone" dataKey="delta" stroke="#FACC15" dot={false} strokeWidth={2} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
