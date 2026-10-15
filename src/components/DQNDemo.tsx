import { useCallback, useEffect, useRef, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine } from "recharts";
import { Play, Pause, StepForward, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { DQNAblation, GWContEnv } from "@/dp/DQNAblation";
import TeX from "@/components/TeX";

const MA = 50;

interface HistPoint {
  ep: number;
  full: number | null;
  noReplay: number | null;
  noTarget: number | null;
}

interface View {
  ep: number;
  history: HistPoint[];
  lastFull: number;
  lastNoReplay: number;
  lastNoTarget: number;
}

function initView(): View {
  return { ep: 0, history: [{ ep: 0, full: null, noReplay: null, noTarget: null }], lastFull: 0, lastNoReplay: 0, lastNoTarget: 0 };
}

export default function DQNDemo() {
  const [speed, setSpeed] = useState(30); // ms / episode
  const [playing, setPlaying] = useState(false);
  const [view, setView] = useState<View>(initView);

  const fullRef = useRef<DQNAblation | null>(null);
  const noRRef = useRef<DQNAblation | null>(null);
  const noTRef = useRef<DQNAblation | null>(null);
  const envRef = useRef<GWContEnv | null>(null);
  const rewardsRef = useRef<{ full: number[]; noR: number[]; noT: number[] }>({ full: [], noR: [], noT: [] });

  const reset = useCallback(() => {
    setPlaying(false);
    const mk = (useReplay: boolean, useTargetNet: boolean) =>
      new DQNAblation(4, 4, {
        lr: 0.001, gamma: 0.99, batchSize: 32, bufferSize: 5000, targetUpdate: 100,
        useReplay, useTargetNet, hidden: [32, 32], epsilonStart: 1.0, epsilonMin: 0.05, epsilonDecay: 0.99,
      });
    fullRef.current = mk(true, true);
    noRRef.current = mk(false, true);
    noTRef.current = mk(true, false);
    envRef.current = new GWContEnv();
    rewardsRef.current = { full: [], noR: [], noT: [] };
    setView(initView());
  }, []);

  useEffect(() => { reset(); }, []);

  const runEpisode = useCallback(() => {
    const full = fullRef.current, noR = noRRef.current, noT = noTRef.current, env = envRef.current;
    if (!full || !noR || !noT || !env) return;
    const one = (dqn: DQNAblation) => {
      let s = env.reset(), total = 0, guard = 0;
      while (guard++ < 250) {
        const a = dqn.selectAction(s, dqn.epsilon);
        const { s2, r, done } = env.step(a);
        dqn.trainStep(s, a, r, s2, done);
        total += r;
        s = s2;
        if (done) break;
      }
      return total;
    };
    const rF = one(full), rR = one(noR), rT = one(noT);
    const rw = rewardsRef.current;
    rw.full.push(rF); rw.noR.push(rR); rw.noT.push(rT);
    const ma = (arr: number[]) => {
      if (arr.length < MA / 2) return null;
      const w = Math.min(MA, arr.length);
      return arr.slice(-w).reduce((x, y) => x + y, 0) / w;
    };
    setView(prev => {
      const ep = prev.ep + 1;
      const history = [...prev.history, { ep, full: ma(rw.full), noReplay: ma(rw.noR), noTarget: ma(rw.noT) }];
      if (history.length > 500) history.shift();
      return { ep, history, lastFull: rF, lastNoReplay: rR, lastNoTarget: rT };
    });
  }, []);

  useEffect(() => {
    if (!playing) return;
    const t = setInterval(runEpisode, speed);
    return () => clearInterval(t);
  }, [playing, speed, runEpisode]);

  const stats = [
    { label: "完整 DQN（replay + target net）", color: "#F0B822", val: view.lastFull },
    { label: "关闭经验回放", color: "#8B5CF6", val: view.lastNoReplay },
    { label: "关闭目标网络（死亡三角发散）", color: "#E0304A", val: view.lastNoTarget },
  ];

  return (
    <div className="rounded-xl border border-border bg-[#0A0A0F] shadow-glow-game overflow-hidden">
      <div className="px-4 py-3 border-b border-border/60 flex flex-wrap items-center justify-between gap-3">
        <div className="text-sm text-muted-foreground">
          <TeX math="L(\theta)=\mathbb{E}\big[(R+\gamma\max_{a'}Q_{\theta^-}(s',a')-Q_\theta(s,a))^2\big]" />
        </div>
        <span className="text-xs font-mono text-text-muted">DQN 消融对照 · GridWorld 连续观测</span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 p-4">
        {/* 消融对照曲线 */}
        <div className="space-y-3">
          <div className="text-xs text-accent-yellow font-semibold mb-1">三组滑动平均回报（同一环境、同一 ε 衰减）</div>
          <div className="h-56 bg-black/30 rounded-lg p-3">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={view.history} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,92,246,0.15)" />
                <XAxis dataKey="ep" stroke="#6B7280" fontSize={10} />
                <YAxis domain={[-30, 12]} stroke="#6B7280" fontSize={10} width={40} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#0A0A0F", border: "1px solid rgba(139,92,246,0.3)", fontSize: 12 }}
                  formatter={(v: number, name: string) => [v?.toFixed(1), name]}
                  labelFormatter={l => `episode ${l}`}
                />
                <ReferenceLine y={0} stroke="#6B7280" strokeOpacity={0.3} />
                <Line type="monotone" dataKey="full" name="完整 DQN" stroke="#F0B822" dot={false} strokeWidth={2.5} isAnimationActive={false} connectNulls />
                <Line type="monotone" dataKey="noReplay" name="关闭 replay" stroke="#8B5CF6" dot={false} strokeWidth={2} isAnimationActive={false} connectNulls opacity={0.85} />
                <Line type="monotone" dataKey="noTarget" name="关闭 target net" stroke="#E0304A" dot={false} strokeWidth={2} isAnimationActive={false} connectNulls />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="text-[11px] text-text-muted leading-relaxed">
            金色完整配方稳步收敛；紫色关 replay 后样本高度相关、早期学习颠簸；红色关 target net 后 bootstrap 目标跟着主网络漂移，
            回报崩塌且无法恢复——这就是「函数逼近 × bootstrap × off-policy」死亡三角中，拔掉任何一根支柱的后果。
          </div>
        </div>

        {/* 控制 + 状态 */}
        <div className="space-y-4">
          <div className="flex items-center gap-2 flex-wrap">
            <Button size="sm" onClick={() => setPlaying(p => !p)}>
              {playing ? <Pause className="w-4 h-4 mr-1" /> : <Play className="w-4 h-4 mr-1" />}
              {playing ? "暂停" : "播放"}
            </Button>
            <Button size="sm" variant="outline" onClick={runEpisode}>
              <StepForward className="w-4 h-4 mr-1" /> 单条轨迹
            </Button>
            <Button size="sm" variant="outline" onClick={reset}>
              <RotateCcw className="w-4 h-4 mr-1" /> 重置
            </Button>
            <span className="text-xs font-mono text-text-muted ml-auto">episode {view.ep}</span>
          </div>

          <div>
            <div className="flex justify-between text-xs text-muted-foreground mb-1"><span>速度 {speed} ms/条</span></div>
            <Slider value={[speed]} min={10} max={200} step={10} onValueChange={([s]) => setSpeed(s)} />
          </div>

          <div className="space-y-2">
            {stats.map(s => (
              <div key={s.label} className="rounded-md bg-black/30 border border-border/50 py-2 px-3 flex items-center justify-between">
                <span className="text-xs" style={{ color: s.color }}>{s.label}</span>
                <span className="font-mono text-sm font-semibold" style={{ color: s.color }}>{s.val.toFixed(1)}</span>
              </div>
            ))}
          </div>

          <div className="text-[11px] text-text-muted leading-relaxed">
            <span className="text-accent-yellow font-semibold">配方的两个零件：</span>
            ① <b>经验回放</b>把顺序到达的强相关样本打散为近似 i.i.d. 的批次（监督学习的训练集回来了）；
            ② <b>目标网络</b>每隔一段时间把主网络"冻结拷贝"给 bootstrap 目标用——靶子不跟着射手动了。
          </div>
        </div>
      </div>
    </div>
  );
}
