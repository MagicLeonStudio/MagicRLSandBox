import { useCallback, useEffect, useRef, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine } from "recharts";
import { Play, Pause, StepForward, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { CartPoleDQN } from "@/dp/CartPoleDQN";
import { CartPoleEngine } from "@/engines/CartPoleEngine";
import { mulberry32 } from "@/lib/rng";
import { Metric } from "@/components/dqn/DQNVisuals";

const GOLD = "#FFD21F";
const PURPLE = "#8B5CF6";
const MELON = "#FF4D6D";
const MAX_TRAIN_STEPS = 20000;
const EVAL_EVERY = 1000;

const GROUPS = [
  { key: "full", label: "完整 DQN", color: GOLD, useReplay: true, useTargetNet: true },
  { key: "noReplay", label: "关闭 Replay", color: PURPLE, useReplay: false, useTargetNet: true },
  { key: "noTarget", label: "关闭 Target Net", color: MELON, useReplay: true, useTargetNet: false },
] as const;

type GroupKey = typeof GROUPS[number]["key"];

interface StressRuntime {
  agent: CartPoleDQN;
  env: CartPoleEngine;
  seed: number;
}

function createRuntimes(seedCount: 1 | 3): Record<GroupKey, StressRuntime[]> {
  const seeds = seedCount === 1 ? [42] : [7, 17, 42];
  const result = {} as Record<GroupKey, StressRuntime[]>;
  for (const group of GROUPS) {
    result[group.key] = seeds.map((seed) => ({
      seed,
      agent: new CartPoleDQN({ seed, useReplay: group.useReplay, useTargetNet: group.useTargetNet }),
      env: new CartPoleEngine(mulberry32(seed + 99)),
    }));
  }
  return result;
}

function stepRuntime(rt: StressRuntime) {
  if (rt.agent.stepCount >= MAX_TRAIN_STEPS) return;
  const state = rt.env.getObservation();
  const action = rt.agent.selectAction(state, rt.agent.epsilon);
  const { state: next, reward, done: failed } = rt.env.step(action);
  const done = failed || rt.env.state.steps >= 200;
  rt.agent.trainStep(state, action, reward, next, done);
  if (done) rt.env.reset();
  if (rt.agent.stepCount % EVAL_EVERY === 0) {
    rt.agent.evaluate(new CartPoleEngine(mulberry32(rt.seed + 9000 + rt.agent.stepCount)), 10);
  }
}

function CartPoleFrame({ engine }: { engine: CartPoleEngine }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (ref.current) engine.render(ref.current);
  });
  return <canvas ref={ref} width={480} height={220} className="w-full rounded-lg border border-[#2A2445] bg-[#07060E]" />;
}

export default function CartPoleStressDemo() {
  const [seedCount, setSeedCount] = useState<1 | 3>(1);
  const [selected, setSelected] = useState<GroupKey>("full");
  const [seedIndex, setSeedIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [stepsPerTick, setStepsPerTick] = useState(8);
  const [, setFrame] = useState(0);
  const runtimesRef = useRef<Record<GroupKey, StressRuntime[]> | null>(null);
  if (!runtimesRef.current) runtimesRef.current = createRuntimes(1);
  const runtimes = runtimesRef.current;

  const reset = useCallback((count: 1 | 3) => {
    setPlaying(false);
    setSeedCount(count);
    setSeedIndex(0);
    runtimesRef.current = createRuntimes(count);
    setFrame((value) => value + 1);
  }, []);

  const tick = useCallback((count = stepsPerTick) => {
    const current = runtimesRef.current;
    if (!current) return;
    for (let i = 0; i < count; i++) {
      for (const group of GROUPS) for (const runtime of current[group.key]) stepRuntime(runtime);
    }
    if (GROUPS.every((group) => current[group.key].every((runtime) => runtime.agent.stepCount >= MAX_TRAIN_STEPS))) {
      setPlaying(false);
    }
    setFrame((value) => value + 1);
  }, [stepsPerTick]);

  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => tick(), 150);
    return () => clearInterval(timer);
  }, [playing, tick]);

  const selectedRuntimes = runtimes[selected];
  const runtime = selectedRuntimes[Math.min(seedIndex, selectedRuntimes.length - 1)];
  const group = GROUPS.find((item) => item.key === selected)!;
  const latest = runtime.agent.evalHistory[runtime.agent.evalHistory.length - 1];
  const checkpoints = Array.from({ length: MAX_TRAIN_STEPS / EVAL_EVERY }, (_, index) => (index + 1) * EVAL_EVERY);
  const chartData = checkpoints.map((step) => {
    const point: Record<string, number> = { step };
    for (const group of GROUPS) {
      const values = runtimes[group.key].map((runtime) => runtime.agent.evalHistory.find((item) => item.step === step)?.avgSteps);
      const available = values.filter((value): value is number => value !== undefined);
      if (available.length) point[group.key] = available.reduce((a, b) => a + b, 0) / available.length;
    }
    return point;
  });

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-2 md:grid-cols-3">
        {GROUPS.map((item) => {
          const histories = runtimes[item.key].map((runtime) => runtime.agent.evalHistory);
          const last = histories.map((history) => history[history.length - 1]).filter(Boolean);
          const mean = last.length ? last.reduce((sum, point) => sum + point.avgSteps, 0) / last.length : null;
          return (
            <button
              key={item.key}
              onClick={() => { setSelected(item.key); setSeedIndex(0); }}
              className="rounded-lg border px-3 py-2 text-left"
              style={{ borderColor: selected === item.key ? item.color : "#2A2445", backgroundColor: selected === item.key ? `${item.color}10` : "#07060E" }}
            >
              <div className="flex items-center justify-between text-xs">
                <span style={{ color: item.color }}>{item.label}</span>
                <span className="font-mono text-[10px] text-[#A79FC7]">{mean === null ? "未评估" : `${mean.toFixed(0)} 步`}</span>
              </div>
            </button>
          );
        })}
      </div>

      <div className="rounded-xl border border-[#2A2445] bg-[#07060E] p-3">
        <div className="flex flex-wrap items-center gap-2">
          <Button size="sm" className="bg-[#FFD21F] text-black hover:bg-[#FFE873]" disabled={runtime.agent.stepCount >= MAX_TRAIN_STEPS} onClick={() => setPlaying((value) => !value)}>
            {playing ? <Pause className="mr-1 h-4 w-4" /> : <Play className="mr-1 h-4 w-4" />}
            {playing ? "暂停" : "播放"}
          </Button>
          <Button size="sm" variant="outline" disabled={playing || runtime.agent.stepCount >= MAX_TRAIN_STEPS} onClick={() => tick(100)}>
            <StepForward className="mr-1 h-4 w-4" />训练 100 步
          </Button>
          <Button size="sm" variant="outline" onClick={() => reset(seedCount)}>
            <RotateCcw className="mr-1 h-4 w-4" />重置
          </Button>
          <div className="ml-auto flex rounded-md border border-[#2A2445] p-0.5 text-[10px]">
            {([1, 3] as const).map((count) => (
              <button
                key={count}
                onClick={() => reset(count)}
                className="rounded px-2 py-1"
                style={{ backgroundColor: seedCount === count ? "#1C1832" : "transparent", color: seedCount === count ? GOLD : "#A79FC7" }}
              >
                {count === 1 ? "单种子" : "三种子"}
              </button>
            ))}
          </div>
        </div>
        <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-[1fr_auto] md:items-center">
          <div>
            <div className="mb-1 flex justify-between text-[10px] text-[#A79FC7]">
              <span>训练速度</span><span className="font-mono">{stepsPerTick} 步 / 帧</span>
            </div>
            <Slider value={[stepsPerTick]} min={1} max={32} step={1} onValueChange={([value]) => setStepsPerTick(value)} />
          </div>
          <div className="font-mono text-[10px] text-[#A79FC7]">STEP {runtime.agent.stepCount} / {MAX_TRAIN_STEPS}</div>
        </div>
        <div className="mt-2 h-1.5 rounded-full bg-[#1C1832]">
          <div className="h-full rounded-full bg-[#FFD21F]" style={{ width: `${runtime.agent.stepCount / MAX_TRAIN_STEPS * 100}%` }} />
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_220px]">
        <div>
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-[10px] text-[#A79FC7]">
            <span>当前环境 · {group.label}</span>
            <span className="flex items-center gap-1">
              {selectedRuntimes.length > 1 && selectedRuntimes.map((item, index) => (
                <button
                  key={item.seed}
                  onClick={() => setSeedIndex(index)}
                  className="rounded border px-1.5 py-0.5 font-mono"
                  style={{ borderColor: seedIndex === index ? group.color : "#2A2445", color: seedIndex === index ? group.color : "#6B6390" }}
                >
                  {item.seed}
                </button>
              ))}
              <span className="ml-1 font-mono">seed {runtime.seed} · ε {runtime.agent.epsilon.toFixed(2)}</span>
            </span>
          </div>
          <CartPoleFrame engine={runtime.env} />
        </div>
        <div className="grid grid-cols-3 gap-3 lg:grid-cols-1">
          <Metric label="最近评估" value={latest ? `${Math.round(latest.successRate * 10)} / 10 成功` : "—"} accent={group.color} />
          <Metric label="平均平衡" value={latest ? `${latest.avgSteps.toFixed(1)} 步` : "—"} accent={group.color} />
          <Metric label="Target 同步" value={runtime.agent.params.useTargetNet ? `STEP ${runtime.agent.lastSyncStep}` : "关闭"} accent={runtime.agent.params.useTargetNet ? PURPLE : MELON} />
        </div>
      </div>

      <div>
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-[10px] text-[#A79FC7]">
          <span>独立贪心评估 · 每 {EVAL_EVERY} 步 × 10 局</span>
          <span>200 步 = 平衡成功</span>
        </div>
        <div className="h-56 rounded-lg bg-[#07060E] p-3">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 6, right: 12, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,92,246,0.12)" />
              <XAxis dataKey="step" stroke="#6B6390" fontSize={10} />
              <YAxis stroke="#6B6390" fontSize={10} width={42} domain={[0, 200]} />
              <Tooltip
                contentStyle={{ backgroundColor: "#0D0B18", border: "1px solid #2A2445", fontSize: 11 }}
                formatter={(value: number, name: string) => [`${value?.toFixed(1)} 步`, name]}
                labelFormatter={(label) => `训练 ${label} 步后评估`}
              />
              <ReferenceLine y={200} stroke={GOLD} strokeDasharray="4 4" strokeOpacity={0.45} />
              <Line type="monotone" dataKey="full" name="完整 DQN" stroke={GOLD} dot={false} strokeWidth={2.4} isAnimationActive={false} connectNulls />
              <Line type="monotone" dataKey="noReplay" name="关闭 replay" stroke={PURPLE} dot={false} strokeWidth={1.8} isAnimationActive={false} connectNulls />
              <Line type="monotone" dataKey="noTarget" name="关闭 target net" stroke={MELON} dot={false} strokeWidth={1.8} isAnimationActive={false} connectNulls />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-2 text-[10px] leading-relaxed text-[#6B6390] md:grid-cols-2">
        <p>
          曲线不是探索期回报，而是每 1000 步后重新进行 10 局 ε=0 评估的平均平衡步数。它直接回答“当前策略是否真的学会了”。
        </p>
        <p>
          完整配方在已验证种子上稳定进入高平衡区；关 target net 常见先学会再崩塌，关 replay 的恢复则更依赖初始化。可用三种子复核随机性。
        </p>
      </div>
    </div>
  );
}
