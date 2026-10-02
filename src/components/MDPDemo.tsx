import { useCallback, useEffect, useMemo, useState } from "react";
import { ArrowUp, ArrowDown, ArrowLeft, ArrowRight, RotateCcw, Dices, Pause } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { GridWorldModel } from "@/dp/GridWorldModel";
import ValueHeatmap from "@/components/ValueHeatmap";
import TeX from "@/components/TeX";

interface StepLog {
  t: number;
  s: number;
  a: number;
  r: number;
  ns: number;
}

const SIZE = 5;
const ACTION_NAMES = ["上", "下", "左", "右"];
const ZERO25 = new Array(SIZE * SIZE).fill(0);
const MAX_STEPS = 100;
const OBSTACLES = [6, 12, 18]; // diagonal scatter across the middle (column #2: red obstacles, bump = -5)

/**
 * MDPDemo —— 读者亲自充当策略 π，在 5x5 GridWorld 中体验五元组：
 * 选动作 a，环境按 P 转移、按 R 发奖励，折扣回报 G_t 随轨迹实时累计。
 * 规则对齐专栏第 2 篇：每步 -0.1，目标 +10，撞障碍 -5 且原地不动，γ 默认 0.99。
 */
export default function MDPDemo() {
  const model = useMemo(() => new GridWorldModel(SIZE, OBSTACLES, -0.1, -5), []);
  const [gamma, setGamma] = useState(0.99);
  const [pos, setPos] = useState(0);
  const [log, setLog] = useState<StepLog[]>([]);
  const [auto, setAuto] = useState(false);

  const done = model.isTerminal(pos);

  const act = useCallback(
    (a: number) => {
      if (model.isTerminal(pos)) return;
      const ns = model.nextState(pos, a);
      const r = model.reward(pos, a, ns);
      setLog(l => [...l, { t: l.length, s: pos, a, r, ns }]);
      setPos(ns);
      if (model.isTerminal(ns)) setAuto(false);
    },
    [model, pos],
  );

  const reset = useCallback(() => {
    setAuto(false);
    setPos(0);
    setLog([]);
  }, []);

  // keyboard control: arrows / WASD
  useEffect(() => {
    const keymap: Record<string, number> = {
      ArrowUp: 0, ArrowDown: 1, ArrowLeft: 2, ArrowRight: 3,
      w: 0, s: 1, a: 2, d: 3,
    };
    const handler = (e: KeyboardEvent) => {
      const a = keymap[e.key];
      if (a === undefined) return;
      e.preventDefault();
      setAuto(false);
      act(a);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [act]);

  // random-policy rollout
  useEffect(() => {
    if (!auto || done || log.length >= MAX_STEPS) {
      if (log.length >= MAX_STEPS) setAuto(false);
      return;
    }
    const timer = setInterval(() => act(Math.floor(Math.random() * 4)), 400);
    return () => clearInterval(timer);
  }, [auto, done, log.length, act]);

  const ret = useMemo(() => log.reduce((g, step) => g + Math.pow(gamma, step.t) * step.r, 0), [log, gamma]);
  const lastR = log.length ? log[log.length - 1].r : null;
  const tail = log.slice(-6).reverse();

  return (
    <div className="rounded-xl border border-border bg-[#0A0A0F] shadow-glow-game overflow-hidden">
      <div className="px-4 py-3 border-b border-border/60 flex flex-wrap items-center justify-between gap-3">
        <span className="text-sm font-semibold">你就是策略 π —— 用方向键 / 按钮走格子</span>
        <div className="text-sm text-muted-foreground">
          <TeX math="G_t=\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}" />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 p-4">
        <div className="flex items-center justify-center bg-black/30 rounded-lg p-4">
          <ValueHeatmap size={SIZE} values={ZERO25} goal={SIZE * SIZE - 1} start={0} showValues={false} agentPos={pos} walls={OBSTACLES} wallTone="danger" />
        </div>

        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <div className="grid grid-cols-3 gap-1 w-fit">
              <span />
              <Button size="sm" variant="outline" onClick={() => act(0)} disabled={done} aria-label="上">
                <ArrowUp className="w-4 h-4" />
              </Button>
              <span />
              <Button size="sm" variant="outline" onClick={() => act(2)} disabled={done} aria-label="左">
                <ArrowLeft className="w-4 h-4" />
              </Button>
              <Button size="sm" variant="outline" onClick={() => act(1)} disabled={done} aria-label="下">
                <ArrowDown className="w-4 h-4" />
              </Button>
              <Button size="sm" variant="outline" onClick={() => act(3)} disabled={done} aria-label="右">
                <ArrowRight className="w-4 h-4" />
              </Button>
            </div>
            <Button size="sm" variant={auto ? "default" : "outline"} onClick={() => setAuto(a => !a)} disabled={done}>
              {auto ? <Pause className="w-4 h-4 mr-1" /> : <Dices className="w-4 h-4 mr-1" />}
              {auto ? "停止随机策略" : "随机策略"}
            </Button>
            <Button size="sm" variant="outline" onClick={reset}>
              <RotateCcw className="w-4 h-4 mr-1" /> 重置
            </Button>
          </div>

          <div className="grid grid-cols-3 gap-2 text-center text-sm">
            <div className="rounded-md bg-black/30 border border-border/50 py-2">
              <div className="text-muted-foreground text-xs">步数 t</div>
              <div className="font-mono text-lg text-accent-yellow">{log.length}</div>
            </div>
            <div className="rounded-md bg-black/30 border border-border/50 py-2">
              <div className="text-muted-foreground text-xs">即时奖励 R_t</div>
              <div className="font-mono text-lg text-accent-purple">
                {lastR === null ? "—" : lastR > 0 ? `+${lastR}` : lastR}
              </div>
            </div>
            <div className="rounded-md bg-black/30 border border-border/50 py-2">
              <div className="text-muted-foreground text-xs">回报 G_0</div>
              <div className="font-mono text-lg text-accent-yellow">{ret.toFixed(3)}</div>
            </div>
          </div>

          <div>
            <div className="flex justify-between text-xs text-muted-foreground mb-1">
              <span>折扣因子 γ = {gamma.toFixed(2)}（拖动后同一条轨迹的回报实时重算）</span>
            </div>
            <Slider value={[gamma]} min={0} max={0.99} step={0.01} onValueChange={([g]) => setGamma(g)} />
          </div>

          <div className="rounded-md bg-black/30 border border-border/50 p-2 min-h-28">
            <div className="text-xs text-muted-foreground mb-1">轨迹（最新在前）：(s, a) → s′，r</div>
            {tail.length === 0 ? (
              <p className="text-xs text-muted-foreground/60 font-mono">尚未行动——白点是智能体，★ 是目标（+10），红色 ✕ 是障碍（撞上 −5），每步 −0.1。</p>
            ) : (
              <div className="space-y-0.5 font-mono text-xs">
                {tail.map(step => (
                  <div key={step.t} className="flex justify-between text-foreground/80">
                    <span>
                      t={step.t} s={step.s} a={ACTION_NAMES[step.a]} → s′={step.ns}
                    </span>
                    <span className={step.r > 0 ? "text-accent-yellow" : step.r <= -5 ? "text-red-400" : "text-muted-foreground"}>
                      r={step.r > 0 ? `+${step.r}` : step.r} · γ^t·r={(Math.pow(gamma, step.t) * step.r).toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {done && (
            <p className="text-sm text-accent-yellow">
              到达目标！本局回报 G_0 = {ret.toFixed(3)}（γ={gamma.toFixed(2)}，{log.length} 步）。点「重置」再来一局。
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
