import { useCallback, useEffect, useRef, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { Play, Pause, StepForward, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { GridWorldModel } from "@/dp/GridWorldModel";
import { makeRng, discountedReturns } from "@/dp/MCTD";
import { SoftmaxPolicy, sampleEpisodeWithPolicy, REINFORCE } from "@/dp/PolicyGrad";
import ValueHeatmap from "@/components/ValueHeatmap";
import TeX from "@/components/TeX";

const WALLS = [6, 7, 11];
const GAMMA = 0.99;
const ACT_NAMES = ["上", "下", "左", "右"];

interface StepDecomp {
  t: number;
  s: number;
  a: number;
  G: number;
  adv: number;
}

interface HistPoint {
  ep: number;
  withB: number;
  withoutB: number;
}

interface View {
  pi: number[];
  path: number[];
  ep: number;
  history: HistPoint[];
  lastSteps: number;
  decomp: StepDecomp[];
  baseline: number;
}

export default function PolicyGradientDemo() {
  const [alpha, setAlpha] = useState(0.05);
  const [speed, setSpeed] = useState(150);
  const [playing, setPlaying] = useState(false);
  const [useBaseline, setUseBaseline] = useState(true);
  const [view, setView] = useState<View>(() => initView());

  const modelRef = useRef<GridWorldModel>(new GridWorldModel(5, WALLS));
  const rfBRef = useRef<REINFORCE | null>(null); // baseline on（主角）
  const rfNRef = useRef<REINFORCE | null>(null); // baseline off（对照）
  const rngBRef = useRef<() => number>(() => 0);
  const rngNRef = useRef<() => number>(() => 0);

  function initView(): View {
    return {
      pi: new Array(25).fill(-1),
      path: [],
      ep: 0,
      history: [{ ep: 0, withB: 0, withoutB: 0 }],
      lastSteps: 0,
      decomp: [],
      baseline: 0,
    };
  }

  const reset = useCallback((a: number) => {
    setPlaying(false);
    const m = new GridWorldModel(5, WALLS);
    modelRef.current = m;
    rfBRef.current = new REINFORCE(new SoftmaxPolicy(m), a, true);
    rfNRef.current = new REINFORCE(new SoftmaxPolicy(m), a, false);
    rngBRef.current = makeRng(31415);
    rngNRef.current = makeRng(31415);
    setView(initView());
  }, []);

  useEffect(() => { reset(alpha); }, []);

  const runEpisode = useCallback(() => {
    const m = modelRef.current, rfB = rfBRef.current, rfN = rfNRef.current;
    if (!rfB || !rfN) return;
    // 同种子轨迹（同参数 θ 下分布相同；两实例 θ 演化路径略异属正常）
    const trajB = sampleEpisodeWithPolicy(m, rfB.policy, rngBRef.current);
    const infoB = rfB.observeEpisode(m, trajB, GAMMA);
    const trajN = sampleEpisodeWithPolicy(m, rfN.policy, rngNRef.current);
    rfN.observeEpisode(m, trajN, GAMMA);

    // 主角（baseline on）的分解视图
    const G = discountedReturns(trajB, GAMMA);
    const b = rfB.baseline;
    const decomp: StepDecomp[] = trajB.actions.map((a, t) => ({
      t, s: trajB.states[t], a, G: G[t], adv: G[t] - b,
    })).slice(-14); // 最近 14 步

    const path = [...trajB.states];
    setView(prev => {
      const ep = prev.ep + 1;
      const history = [...prev.history, { ep, withB: infoB.steps, withoutB: trajN.states.length - 1 }];
      if (history.length > 400) history.shift();
      return {
        pi: rfB.policy.greedyPolicy(m),
        path,
        ep, history,
        lastSteps: infoB.steps,
        decomp, baseline: b,
      };
    });
  }, []);

  useEffect(() => {
    if (!playing) return;
    const t = setInterval(runEpisode, speed);
    return () => clearInterval(t);
  }, [playing, speed, runEpisode]);

  const actColor = (adv: number) => (adv >= 0 ? "#F0B822" : "#E0304A");

  return (
    <div className="rounded-xl border border-border bg-[#0A0A0F] shadow-glow-game overflow-hidden">
      <div className="px-4 py-3 border-b border-border/60 flex flex-wrap items-center justify-between gap-3">
        <div className="text-sm text-muted-foreground">
          <TeX math="\theta\leftarrow\theta+\alpha\,\nabla\log\pi_\theta(A_t|S_t)\cdot\big(G_t-b\big)" />
        </div>
        <span className="text-xs font-mono text-text-muted">REINFORCE · GridWorld 5×5</span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 p-4">
        {/* 策略地图 + 最近轨迹 */}
        <div className="space-y-3">
          <div>
            <div className="text-xs text-accent-yellow font-semibold mb-1">
              策略地图（贪心动作箭头 · 随训练演化）{!useBaseline && " · baseline 关闭"}
            </div>
            <div className="flex justify-center bg-black/30 rounded-lg p-3">
              <ValueHeatmap
                size={5}
                values={view.pi.map((a, s) => (a >= 0 ? rfBRef.current?.policy.probs(s)[a] ?? 0 : 0))}
                policy={view.pi}
                walls={WALLS}
                goal={24}
                highlightPath={view.path}
                showValues={false}
              />
            </div>
          </div>
          <div className="text-[11px] text-text-muted leading-relaxed">
            白色描边高亮是最近一条采样轨迹。注意箭头如何从"四处乱指"收敛为指向右下角的最优路径——
            这不是查价值表得到的，而是每一步动作的 log 概率被回报持续推高的结果。
          </div>
        </div>

        {/* 控制 + 曲线 + 分解 */}
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

          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant={useBaseline ? "default" : "outline"}
              onClick={() => {
                setUseBaseline(!useBaseline);
                reset(alpha);
              }}
            >
              baseline {useBaseline ? "开" : "关"}（当前显示：{useBaseline ? "开" : "关"}组）
            </Button>
            <span className="text-xs text-text-muted">b = {view.baseline.toFixed(2)}</span>
          </div>

          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-xs text-muted-foreground mb-1"><span>步长 α = {alpha.toFixed(3)}</span></div>
              <Slider value={[alpha]} min={0.005} max={0.2} step={0.005} onValueChange={([a]) => { setAlpha(a); reset(a); }} />
            </div>
            <div>
              <div className="flex justify-between text-xs text-muted-foreground mb-1"><span>速度 {speed} ms/条</span></div>
              <Slider value={[speed]} min={40} max={600} step={20} onValueChange={([s]) => setSpeed(s)} />
            </div>
          </div>

          <div className="h-40">
            <div className="text-xs text-muted-foreground mb-1">每条轨迹的步数（越低越好；两条曲线均为同种子独立训练）</div>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={view.history} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,92,246,0.15)" />
                <XAxis dataKey="ep" stroke="#6B7280" fontSize={10} />
                <YAxis domain={[0, 80]} stroke="#6B7280" fontSize={10} width={36} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#0A0A0F", border: "1px solid rgba(139,92,246,0.3)", fontSize: 12 }}
                  formatter={(v: number, name: string) => [v, name]}
                  labelFormatter={l => `episode ${l}`}
                />
                <Line type="monotone" dataKey="withB" name="有 baseline" stroke="#F0B822" dot={false} strokeWidth={2} isAnimationActive={false} />
                <Line type="monotone" dataKey="withoutB" name="无 baseline" stroke="#8B5CF6" dot={false} strokeWidth={2} isAnimationActive={false} opacity={0.8} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* 单条轨迹的梯度分解 */}
          <div>
            <div className="text-xs text-muted-foreground mb-1">最近轨迹的逐步分解（adv = G − b：金色 = 强化该动作，红色 = 抑制）</div>
            <div className="rounded-md bg-black/40 border border-border/50 p-2 max-h-40 overflow-y-auto space-y-1">
              {view.decomp.length === 0 && (
                <div className="text-xs text-text-muted text-center py-2">运行一条轨迹后，这里逐步展示 ∇log π · (G − b) 的贡献</div>
              )}
              {view.decomp.map((d) => (
                <div key={d.t} className="flex items-center gap-2 text-[11px] font-mono">
                  <span className="text-text-muted w-10">t={d.t}</span>
                  <span className="text-text-secondary w-20">s={d.s} · {ACT_NAMES[d.a]}</span>
                  <div className="flex-1 h-2 rounded-full bg-bg-input overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${Math.min(100, Math.abs(d.adv) / 6 * 100)}%`,
                        backgroundColor: actColor(d.adv),
                      }}
                    />
                  </div>
                  <span className="w-14 text-right" style={{ color: actColor(d.adv) }}>
                    {d.adv >= 0 ? "+" : ""}{d.adv.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
