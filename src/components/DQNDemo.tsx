import { useCallback, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { Play, Pause, StepForward, RotateCcw, Eye, Database, Copy, Lock, ScanEye } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { PixelDQN, PixelGridEnv } from "@/dp/PixelDQN";
import type { PixelExperience } from "@/dp/PixelDQN";
import TeX from "@/components/TeX";
import CartPoleStressDemo from "@/components/dqn/CartPoleStressDemo";
import { PixelFrame, FeatureGrid, DenseBars, QValueBars, Metric, ACTION_ARROWS } from "@/components/dqn/DQNVisuals";

const GOLD = "#FFD21F";
const PURPLE = "#8B5CF6";
const MELON = "#FF4D6D";
const MA_WINDOW = 20;

const GROUPS = [
  { key: "full", label: "完整 DQN", sub: "replay + target net", color: GOLD, useReplay: true, useTargetNet: true },
  { key: "noReplay", label: "关闭 Replay", sub: "在线顺序样本", color: PURPLE, useReplay: false, useTargetNet: true },
  { key: "noTarget", label: "关闭 Target Net", sub: "目标跟随主网络", color: MELON, useReplay: true, useTargetNet: false },
] as const;

type GroupKey = typeof GROUPS[number]["key"];
type LayoutMode = "fixed" | "randomStart";

interface Runtime {
  agent: PixelDQN;
  env: PixelGridEnv;
  obs: Float32Array;
  returns: number[];
  epReturn: number;
  lastReturn: number | null;
  lastAction: number;
  lastReward: number;
}

interface EvalState {
  key: GroupKey;
  env: PixelGridEnv;
  obs: Float32Array;
  ret: number;
  steps: number;
  done: boolean;
  lastAction: number;
  lastReward: number;
}

function createRuntimes(layout: LayoutMode): Record<GroupKey, Runtime> {
  const probeEnv = new PixelGridEnv(20241015);
  const probe = probeEnv.obs();
  const result = {} as Record<GroupKey, Runtime>;
  for (const group of GROUPS) {
    const agent = new PixelDQN({ seed: 7, useReplay: group.useReplay, useTargetNet: group.useTargetNet });
    agent.setProbe(probe);
    const env = new PixelGridEnv(99);
    env.layout = layout;
    result[group.key] = {
      agent,
      env,
      obs: env.obs(),
      returns: [],
      epReturn: 0,
      lastReturn: null,
      lastAction: -1,
      lastReward: 0,
    };
  }
  return result;
}

function stepRuntime(rt: Runtime) {
  const action = rt.agent.selectAction(rt.obs, rt.agent.epsilon, rt.env.validActions());
  const { s2, r, done } = rt.env.step(action);
  rt.agent.trainStep(rt.obs, action, r, s2, done, rt.env.episode, rt.env.steps, rt.env.validActions());
  rt.obs = s2;
  rt.epReturn += r;
  rt.lastAction = action;
  rt.lastReward = r;
  if (done) {
    rt.returns.push(rt.epReturn);
    rt.lastReturn = rt.epReturn;
    rt.epReturn = 0;
    rt.agent.finishEpisode();
    rt.obs = rt.env.reset();
  }
}

function movingAverage(values: number[], index: number): number | null {
  if (index < 4 || index >= values.length) return null;
  const start = Math.max(0, index - MA_WINDOW + 1);
  const slice = values.slice(start, index + 1);
  return slice.reduce((a, b) => a + b, 0) / slice.length;
}

function Panel({
  index,
  title,
  subtitle,
  children,
  accent = GOLD,
  className = "",
}: {
  index: string;
  title: string;
  subtitle?: string;
  children: ReactNode;
  accent?: string;
  className?: string;
}) {
  return (
    <section className={`rounded-xl border border-[#2A2445] bg-[#0D0B18] ${className}`}>
      <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-[#2A2445]/80 px-4 py-3">
        <div className="flex items-baseline gap-2">
          <span className="font-mono text-[10px] tracking-[0.16em]" style={{ color: accent }}>{index}</span>
          <h3 className="text-sm font-semibold text-[#F1EEFB]">{title}</h3>
        </div>
        {subtitle && <span className="text-[10px] text-[#6B6390]">{subtitle}</span>}
      </div>
      <div className="p-4">{children}</div>
    </section>
  );
}

function ExperienceThumb({ exp, accent, muted = false }: { exp: PixelExperience; accent: string; muted?: boolean }) {
  return (
    <div
      className="shrink-0 rounded-md border bg-[#07060E] p-1"
      style={{ borderColor: muted ? "#2A2445" : accent, opacity: muted ? 0.55 : 1 }}
      title={`episode ${exp.episode} · step ${exp.step} · action ${ACTION_ARROWS[exp.a]} · r ${exp.r.toFixed(2)}`}
    >
      <PixelFrame data={exp.s} size={34} />
      <div className="mt-1 flex items-center justify-between font-mono text-[8px] text-[#A79FC7]">
        <span>E{exp.episode}:{exp.step}</span>
        <span style={{ color: exp.r > 0 ? GOLD : undefined }}>{ACTION_ARROWS[exp.a]}</span>
      </div>
    </div>
  );
}

export default function DQNDemo() {
  const [layout, setLayout] = useState<LayoutMode>("fixed");
  const [selected, setSelected] = useState<GroupKey>("full");
  const [playing, setPlaying] = useState(false);
  const [stepsPerTick, setStepsPerTick] = useState(4);
  const [, setFrame] = useState(0);
  const [evalState, setEvalState] = useState<EvalState | null>(null);
  const [resultTab, setResultTab] = useState<"cartpole" | "pixel">("cartpole");
  const runtimesRef = useRef<Record<GroupKey, Runtime> | null>(null);
  if (!runtimesRef.current) runtimesRef.current = createRuntimes("fixed");
  const runtimes = runtimesRef.current;

  const reset = useCallback((mode: LayoutMode) => {
    setPlaying(false);
    setEvalState(null);
    setLayout(mode);
    runtimesRef.current = createRuntimes(mode);
    setFrame((v) => v + 1);
  }, []);

  const tick = useCallback((count = stepsPerTick) => {
    const current = runtimesRef.current;
    if (!current) return;
    for (let i = 0; i < count; i++) {
      for (const group of GROUPS) stepRuntime(current[group.key]);
    }
    setFrame((v) => v + 1);
  }, [stepsPerTick]);

  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => tick(), 120);
    return () => clearInterval(timer);
  }, [playing, tick]);

  const startEvaluation = () => {
    if (evalState && !evalState.done) {
      setEvalState(null);
      return;
    }
    setPlaying(false);
    const rt = runtimesRef.current?.[selected];
    if (!rt) return;
    const env = new PixelGridEnv(5000 + rt.agent.stepCount);
    env.layout = layout;
    setEvalState({
      key: selected,
      env,
      obs: env.obs(),
      ret: 0,
      steps: 0,
      done: false,
      lastAction: -1,
      lastReward: 0,
    });
  };

  const evalActive = evalState !== null && !evalState.done;
  useEffect(() => {
    if (!evalActive) return;
    const timer = setInterval(() => {
      setEvalState((prev) => {
        if (!prev || prev.done) return prev;
        const rt = runtimesRef.current?.[prev.key];
        if (!rt) return prev;
        const action = rt.agent.selectAction(prev.obs, 0, prev.env.validActions());
        const { s2, r, done } = prev.env.step(action);
        return {
          ...prev,
          obs: s2,
          ret: prev.ret + r,
          steps: prev.steps + 1,
          done,
          lastAction: action,
          lastReward: r,
        };
      });
    }, 220);
    return () => clearInterval(timer);
  }, [evalActive]);

  const rt = runtimes[selected];
  const group = GROUPS.find((item) => item.key === selected)!;
  const displayObs = evalState?.obs ?? rt.obs;
  const inspection = rt.agent.inspect(displayObs);
  const referenceQ = rt.agent.referenceQ(displayObs);
  const displayValid = (evalState ? evalState.env : rt.env).validActions();
  const displayAction = evalState ? evalState.lastAction : rt.lastAction;
  const displayReward = evalState ? evalState.lastReward : rt.lastReward;
  const referenceColor = rt.agent.params.useTargetNet ? PURPLE : MELON;
  const batch = rt.agent.lastBatch;
  const batchSources = new Set(batch.map((exp) => exp.episode)).size;
  const sortedIds = batch.map((exp) => exp.id).sort((a, b) => a - b);
  const adjacentPairs = sortedIds.slice(1).filter((id, i) => id === sortedIds[i] + 1).length;
  const recentExperiences = rt.agent.buffer.slice(-14);
  const train = rt.agent.lastTrain;
  const syncRecent = rt.agent.params.useTargetNet && rt.agent.lastSyncStep > 0
    && rt.agent.stepCount - rt.agent.lastSyncStep < stepsPerTick;

  const episodeCount = Math.max(...GROUPS.map((item) => runtimes[item.key].returns.length));
  const rawHistory = Array.from({ length: episodeCount }, (_, index) => ({
    ep: index + 1,
    full: movingAverage(runtimes.full.returns, index),
    noReplay: movingAverage(runtimes.noReplay.returns, index),
    noTarget: movingAverage(runtimes.noTarget.returns, index),
  }));
  const stride = Math.max(1, Math.ceil(rawHistory.length / 360));
  const history = rawHistory.filter((_, index) => index % stride === 0 || index === rawHistory.length - 1);

  return (
    <div id="dqn-mechanism-lab" className="overflow-hidden rounded-xl border border-[#2A2445] bg-[#07060E] text-[#F1EEFB] shadow-[0_0_24px_rgba(139,92,246,0.12)]">
      <div className="border-b border-[#2A2445] bg-[#0D0B18] px-4 py-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="font-mono text-[10px] tracking-[0.18em] text-[#8B5CF6]">PIXEL DQN · MECHANISM LAB</div>
            <div className="mt-1 text-sm font-semibold">像素如何经过 Deep Q Network 变成动作</div>
          </div>
          <div className="hidden md:block text-xs text-[#A79FC7]">
            <TeX math="y=r+\gamma\max_{a'}Q_{\theta^-}(s',a')" />
          </div>
        </div>
      </div>

      <div className="space-y-4 p-3 md:p-4">
        <div className="grid grid-cols-1 gap-2 md:grid-cols-3">
          {GROUPS.map((item) => {
            const runtime = runtimes[item.key];
            const ma = movingAverage(runtime.returns, runtime.returns.length - 1);
            const active = selected === item.key;
            return (
              <button
                key={item.key}
                onClick={() => {
                  if (!evalState || evalState.done) {
                    setSelected(item.key);
                    setEvalState(null);
                  }
                }}
                className="rounded-lg border px-3 py-2.5 text-left transition-colors"
                style={{
                  borderColor: active ? item.color : "#2A2445",
                  backgroundColor: active ? `${item.color}10` : "#0D0B18",
                  boxShadow: active ? `0 0 18px ${item.color}18` : undefined,
                }}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs font-semibold" style={{ color: item.color }}>{item.label}</span>
                  <span className="font-mono text-[10px] text-[#6B6390]">EP {runtime.returns.length}</span>
                </div>
                <div className="mt-1 flex items-center justify-between text-[10px] text-[#A79FC7]">
                  <span>{item.sub}</span>
                  <span className="font-mono">MA20 {ma === null ? "—" : ma.toFixed(2)}</span>
                </div>
              </button>
            );
          })}
        </div>

        <div className="rounded-xl border border-[#2A2445] bg-[#0D0B18] p-3">
          <div className="flex flex-wrap items-center gap-2">
            <Button size="sm" className="bg-[#FFD21F] text-black hover:bg-[#FFE873]" onClick={() => { setEvalState(null); setPlaying((value) => !value); }}>
              {playing ? <Pause className="mr-1 h-4 w-4" /> : <Play className="mr-1 h-4 w-4" />}
              {playing ? "暂停" : "播放"}
            </Button>
            <Button size="sm" variant="outline" disabled={playing} onClick={() => { setEvalState(null); tick(1); }}>
              <StepForward className="mr-1 h-4 w-4" />单步训练
            </Button>
            <Button size="sm" variant="outline" onClick={startEvaluation}>
              <Eye className="mr-1 h-4 w-4" />{evalActive ? "停止评估" : "评估策略"}
            </Button>
            <Button size="sm" variant="outline" onClick={() => reset(layout)}>
              <RotateCcw className="mr-1 h-4 w-4" />重置
            </Button>
            <span className="ml-auto font-mono text-[10px] text-[#6B6390]">
              STEP {rt.agent.stepCount} · ε {rt.agent.epsilon.toFixed(2)}
            </span>
          </div>
          <div className="mt-3 grid grid-cols-1 gap-4 md:grid-cols-[1fr_auto] md:items-center">
            <div>
              <div className="mb-1 flex justify-between text-[10px] text-[#A79FC7]">
                <span>训练速度</span><span className="font-mono">{stepsPerTick} 步 / 帧</span>
              </div>
              <Slider value={[stepsPerTick]} min={1} max={16} step={1} onValueChange={([value]) => setStepsPerTick(value)} />
            </div>
            <div className="flex rounded-md border border-[#2A2445] p-0.5 text-[10px]">
              {(["fixed", "randomStart"] as LayoutMode[]).map((mode) => (
                <button
                  key={mode}
                  onClick={() => reset(mode)}
                  className="rounded px-2.5 py-1.5"
                  style={{ backgroundColor: layout === mode ? "#1C1832" : "transparent", color: layout === mode ? GOLD : "#A79FC7" }}
                >
                  {mode === "fixed" ? "固定起终点" : "随机起点"}
                </button>
              ))}
            </div>
          </div>
          {evalState && (
            <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-[#2A2445] pt-2 font-mono text-[10px] text-[#A79FC7]">
              <span className="text-[#3EC6F0]">ε=0 EVALUATION</span>
              <span>return {evalState.ret.toFixed(2)}</span>
              <span>steps {evalState.steps}</span>
              {evalState.done && (
                <span style={{ color: evalState.env.pos === evalState.env.goal ? GOLD : MELON }}>
                  {evalState.env.pos === evalState.env.goal ? "到达目标" : "超时未到达"}
                </span>
              )}
            </div>
          )}
        </div>

        <Panel index="01" title="看见 Deep：像素 → 卷积特征 → Q 值 → 动作" subtitle="所有特征图均来自当前网络的真实前向传播">
          <div className="grid grid-cols-1 gap-5 xl:grid-cols-[220px_minmax(0,1fr)_220px]">
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <div className="rounded-lg border border-[#3D3463] bg-[#07060E] p-2 shadow-[0_0_20px_rgba(255,210,31,0.08)]">
                  <PixelFrame data={displayObs} size={168} />
                </div>
                <div className="space-y-2 text-[10px] text-[#A79FC7]">
                  <div><span className="mr-1 inline-block h-2 w-2 bg-[#FFD21F]" />智能体</div>
                  <div><span className="mr-1 inline-block h-2 w-2 bg-[#8B5CF6]" />目标</div>
                  <div className="pt-1 font-mono text-[#6B6390]">12 × 12 × 3<br />随机底纹</div>
                </div>
              </div>
              <div className="grid grid-cols-3 gap-2 border-t border-[#2A2445] pt-3">
                <Metric label="动作" value={displayAction >= 0 ? ACTION_ARROWS[displayAction] : "—"} accent={group.color} />
                <Metric label="即时奖励" value={displayReward.toFixed(2)} accent={displayReward >= 0 ? GOLD : MELON} />
                <Metric label="输入" value="PIXELS" accent="#3EC6F0" />
              </div>
              <p className="text-[10px] leading-relaxed text-[#6B6390]">
                网络只读取像素，不读取格子坐标。金块与紫块的位置必须由视觉特征自己推断。
              </p>
            </div>

            <div className="space-y-4 border-[#2A2445] xl:border-x xl:px-5">
              <div>
                <div className="mb-2 flex items-center justify-between text-[10px]">
                  <span className="font-mono text-[#A79FC7]">CONV 1 · 4 × 5 × 5</span>
                  <span className="text-[#6B6390]">3×3 / stride 2 / ReLU</span>
                </div>
                <FeatureGrid data={inspection.activations.conv1} side={5} filters={4} accent={group.color} size={46} />
              </div>
              <div>
                <div className="mb-2 flex items-center justify-between text-[10px]">
                  <span className="font-mono text-[#A79FC7]">CONV 2 · 8 × 2 × 2</span>
                  <span className="text-[#6B6390]">3×3 / stride 2 / ReLU</span>
                </div>
                <FeatureGrid data={inspection.activations.conv2} side={2} filters={8} accent={group.color} size={38} />
              </div>
              <div>
                <div className="mb-1 flex items-center justify-between text-[10px]">
                  <span className="font-mono text-[#A79FC7]">DENSE · 64</span>
                  <span className="text-[#6B6390]">flatten 32 → 64</span>
                </div>
                <DenseBars data={inspection.activations.dense} accent={group.color} />
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <div className="mb-2 flex items-center justify-between text-[10px]">
                  <span className="font-mono" style={{ color: group.color }}>ONLINE Qθ</span>
                  <span className="text-[#6B6390]">四动作价值</span>
                </div>
                <QValueBars q={inspection.q} accent={group.color} valid={displayValid} />
              </div>
              <div className="border-t border-[#2A2445] pt-3">
                <div className="mb-2 flex items-center justify-between text-[10px]">
                  <span className="font-mono" style={{ color: referenceColor }}>
                    {rt.agent.params.useTargetNet ? "TARGET Qθ⁻" : "REFERENCE = ONLINE"}
                  </span>
                  {rt.agent.params.useTargetNet && <Lock className="h-3 w-3 text-[#8B5CF6]" />}
                </div>
                <QValueBars q={referenceQ} accent={referenceColor} valid={displayValid} compact />
              </div>
              <div className="border-t border-[#2A2445] pt-3 font-mono text-[10px] leading-5 text-[#A79FC7]">
                <div className="mb-1 font-sans text-[#6B6390]">最近训练样本</div>
                <div>y = {train ? train.target.toFixed(3) : "—"}</div>
                <div>Qθ(s,a) = {train ? train.prediction.toFixed(3) : "—"}</div>
                <div style={{ color: train && Math.abs(train.tdError) > 0.5 ? MELON : GOLD }}>
                  TD error = {train ? train.tdError.toFixed(3) : "—"}
                </div>
              </div>
            </div>
          </div>
          <div className="mt-4 border-t border-[#2A2445] pt-3 text-center font-mono text-[10px] text-[#6B6390]">
            12×12×3 → Conv 4/s2 → Conv 8/s2 → Dense 64 → Q(up, down, left, right)
          </div>
        </Panel>

        <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
          <Panel index="02" title="看见 Replay：训练样本从哪里来" subtitle="经验胶片与实际 batch 同步更新" accent={PURPLE}>
            <div className="space-y-4">
              <div className="grid grid-cols-3 gap-3">
                <Metric label={rt.agent.params.useReplay ? "回放池" : "经验缓存（可视化）"} value={`${rt.agent.buffer.length} / ${rt.agent.params.bufferSize}`} accent={PURPLE} />
                <Metric label="本批大小" value={rt.agent.params.useReplay ? String(batch.length) : "1（在线）"} accent={group.color} />
                <Metric label="来源轨迹" value={`${batchSources} 条`} accent={group.color} />
              </div>

              <div>
                <div className="mb-2 flex items-center gap-1.5 text-[10px] text-[#A79FC7]">
                  <Database className="h-3.5 w-3.5 text-[#8B5CF6]" />
                  {rt.agent.params.useReplay ? "最新经验流 → 回放池" : "最新经验流（不用于回放训练）"}
                </div>
                <div className="flex gap-1.5 overflow-x-auto pb-2">
                  {recentExperiences.length === 0 && <span className="text-[10px] text-[#6B6390]">开始训练后，真实转移会出现在这里</span>}
                  {recentExperiences.map((exp) => <ExperienceThumb key={exp.id} exp={exp} accent={PURPLE} muted />)}
                </div>
              </div>

              <div className="border-t border-[#2A2445] pt-3">
                <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-[10px]">
                  <span style={{ color: group.color }}>
                    {rt.agent.params.useReplay ? `随机抽中的 ${batch.length} 条训练样本` : "顺序到达：只用最新 1 条"}
                  </span>
                  <span className="font-mono text-[#6B6390]">
                    {batch.length > 1 ? `时间相邻对 ${adjacentPairs} / ${batch.length - 1}` : "无随机打散"}
                  </span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {batch.length === 0 && <span className="text-[10px] text-[#6B6390]">回放池预热中，尚未形成 batch</span>}
                  {batch.map((exp, index) => <ExperienceThumb key={`${exp.id}-${index}`} exp={exp} accent={group.color} />)}
                </div>
                <p className="mt-3 text-[10px] leading-relaxed text-[#6B6390]">
                  {rt.agent.params.useReplay
                    ? "金色描边的样本来自不同时间与轨迹：监督学习所需的“乱序训练集”在这里真实形成。"
                    : "关闭 replay 后，网络每步只学习刚到达的转移；连续画面的内容高度相似，更新方向更容易互相拉扯。"}
                </p>
              </div>
            </div>
          </Panel>

          <Panel index="03" title="看见 Target Net：靶子什么时候才动" subtitle="固定探针隔离“目标冻结”这一现象" accent={referenceColor}>
            <div className="space-y-4">
              <div className="grid grid-cols-3 gap-3">
                <Metric
                  label="参数同步"
                  value={rt.agent.params.useTargetNet ? `${rt.agent.syncCountdown()} 步后` : "每步跟随"}
                  accent={syncRecent ? GOLD : referenceColor}
                />
                <Metric label="上次同步" value={rt.agent.lastSyncStep ? `STEP ${rt.agent.lastSyncStep}` : "初始拷贝"} accent={referenceColor} />
                <Metric label="探针动作" value="→" accent={GOLD} />
              </div>

              <div>
                <div className="mb-2 flex items-center justify-between text-[10px]">
                  <span className="flex items-center gap-1.5 text-[#A79FC7]">
                    <ScanEye className="h-3.5 w-3.5" />同一像素探针的 Q 值轨迹
                  </span>
                  <span className="font-mono text-[#6B6390]">online / reference</span>
                </div>
                <div className="h-36 rounded-lg bg-[#07060E] p-2">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={rt.agent.probeHistory} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,92,246,0.12)" />
                      <XAxis dataKey="step" stroke="#6B6390" fontSize={9} />
                      <YAxis stroke="#6B6390" fontSize={9} width={44} domain={["auto", "auto"]} />
                      <Tooltip
                        contentStyle={{ backgroundColor: "#0D0B18", border: "1px solid #2A2445", fontSize: 11 }}
                        formatter={(value: number, name: string) => [value?.toFixed(3), name]}
                      />
                      <Line type="monotone" dataKey="online" name="Online Qθ" stroke={GOLD} dot={false} strokeWidth={2} isAnimationActive={false} />
                      <Line
                        type="stepAfter"
                        dataKey="reference"
                        name={rt.agent.params.useTargetNet ? "Target Qθ⁻" : "Reference = Online"}
                        stroke={referenceColor}
                        dot={false}
                        strokeWidth={2}
                        isAnimationActive={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <p className="mt-2 text-[10px] leading-relaxed text-[#6B6390]">
                  {rt.agent.params.useTargetNet
                    ? "金线持续变化，紫色参照只在参数同步时跳变：这就是“靶子不跟着射手动”。"
                    : "关闭目标网络后，参照与在线网络同源，bootstrap 的靶子随每次更新一起移动。"}
                </p>
              </div>

              <div className="border-t border-[#2A2445] pt-3">
                <div className="mb-2 flex items-center gap-1.5 text-[10px] text-[#A79FC7]">
                  <Copy className="h-3.5 w-3.5" style={{ color: syncRecent ? GOLD : referenceColor }} />
                  {syncRecent ? "刚刚发生 θ⁻ ← θ 参数拷贝" : "最近样本的目标值分解"}
                </div>
                <div className="rounded-lg bg-[#07060E] px-3 py-2 font-mono text-[11px] leading-6 text-[#A79FC7]">
                  <div>r = {train ? train.reward.toFixed(3) : "—"}</div>
                  <div>max Qref(s′) = {train ? train.maxNext.toFixed(3) : "—"}</div>
                  <div style={{ color: GOLD }}>y = r + 0.95 × maxQ = {train ? train.target.toFixed(3) : "—"}</div>
                </div>
              </div>
            </div>
          </Panel>
        </div>

        <Panel index="04" title="结果区：回报曲线与 CartPole 压力测试" subtitle="训练回报与独立贪心评估分开" accent={group.color}>
          <div className="mb-4 flex rounded-lg border border-[#2A2445] p-1 text-[11px]">
            {(["cartpole", "pixel"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setResultTab(tab)}
                className="flex-1 rounded-md px-3 py-2"
                style={{
                  backgroundColor: resultTab === tab ? "#1C1832" : "transparent",
                  color: resultTab === tab ? GOLD : "#A79FC7",
                }}
              >
                {tab === "cartpole" ? "CartPole 稳定性压力测试" : "GridWorld 三组回报"}
              </button>
            ))}
          </div>
          <div style={{ display: resultTab === "cartpole" ? "block" : "none" }}>
            <CartPoleStressDemo />
          </div>
          <div style={{ display: resultTab === "pixel" ? "block" : "none" }}>
          <div className="h-56 rounded-lg bg-[#07060E] p-3">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 6, right: 12, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,92,246,0.12)" />
                <XAxis dataKey="ep" stroke="#6B6390" fontSize={10} />
                <YAxis stroke="#6B6390" fontSize={10} width={46} domain={["auto", "auto"]} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#0D0B18", border: "1px solid #2A2445", fontSize: 11 }}
                  formatter={(value: number, name: string) => [value?.toFixed(2), name]}
                  labelFormatter={(label) => `episode ${label}`}
                />
                <Line type="monotone" dataKey="full" name="完整 DQN" stroke={GOLD} dot={false} strokeWidth={2.4} isAnimationActive={false} connectNulls />
                <Line type="monotone" dataKey="noReplay" name="关闭 replay" stroke={PURPLE} dot={false} strokeWidth={1.8} isAnimationActive={false} connectNulls />
                <Line type="monotone" dataKey="noTarget" name="关闭 target net" stroke={MELON} dot={false} strokeWidth={1.8} isAnimationActive={false} connectNulls />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-3 grid grid-cols-1 gap-2 text-[10px] leading-relaxed text-[#6B6390] md:grid-cols-2">
            <p>
              点击上方任一组可查看其内部机制。训练曲线使用探索期回报；要看最终策略，请切换到“评估策略”，以 ε=0 观察实际导航。
            </p>
            <p>
              这是计算量友好的小型视觉任务，三组都可能最终学会，不把单次发散当作必然结论。重点对照：batch 的时间相关性、目标参照的漂移，以及学习过程是否稳定。
            </p>
          </div>
          </div>
        </Panel>
      </div>
    </div>
  );
}
