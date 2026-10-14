import { useMemo, useState } from "react";
import { ComposedChart, Line, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine, Bar, Cell } from "recharts";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import TeX from "@/components/TeX";

const GAMMA = 0.99;

/** Tab 1: PPO clip 目标函数 L^CLIP = min(r·A, clip(r)·A) */
function ClipView() {
  const [eps, setEps] = useState(0.2);
  const [advSign, setAdvSign] = useState<1 | -1>(1);

  const data = useMemo(() => {
    const A = advSign;
    const out: { r: number; cpi: number; clip: number; gap: [number, number] | null }[] = [];
    for (let i = 0; i <= 100; i++) {
      const r = i / 50; // 0 ~ 2
      const clipR = Math.min(1 + eps, Math.max(1 - eps, r));
      const cpi = r * A;
      const clip = Math.min(cpi, clipR * A);
      // clipped-away region: strictly between the clipped (lower) and unclipped (upper) curves
      const gap: [number, number] | null = cpi - clip > 1e-9 ? [clip, cpi] : null;
      out.push({ r, cpi, clip, gap });
    }
    return out;
  }, [eps, advSign]);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">优势符号 A：</span>
          <Button size="sm" variant={advSign === 1 ? "default" : "outline"} onClick={() => setAdvSign(1)}>
            A &gt; 0（该强化）
          </Button>
          <Button size="sm" variant={advSign === -1 ? "default" : "outline"} onClick={() => setAdvSign(-1)}>
            A &lt; 0（该抑制）
          </Button>
        </div>
        <div className="flex-1 min-w-[200px]">
          <div className="flex justify-between text-xs text-muted-foreground mb-1"><span>裁剪范围 ε = {eps.toFixed(2)}</span></div>
          <Slider value={[eps]} min={0.05} max={0.3} step={0.01} onValueChange={([e]) => setEps(e)} />
        </div>
      </div>

      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,92,246,0.15)" />
            <XAxis dataKey="r" stroke="#6B7280" fontSize={10} type="number" domain={[0, 2]} tickCount={9}
              label={{ value: "重要性采样比 r", position: "insideBottomRight", offset: -2, fontSize: 10, fill: "#6B7280" }} />
            <YAxis stroke="#6B7280" fontSize={10} domain={[-2.2, 2.2]} />
            <Tooltip
              contentStyle={{ backgroundColor: "#0A0A0F", border: "1px solid rgba(139,92,246,0.3)", fontSize: 12 }}
              formatter={(v: number | [number, number], name: string) =>
                Array.isArray(v) ? [`${v[0].toFixed(2)} ~ ${v[1].toFixed(2)}`, name] : [v.toFixed(3), name]}
              labelFormatter={l => `r = ${l}`}
            />
            <ReferenceLine x={1} stroke="#6B7280" strokeDasharray="2 2" strokeOpacity={0.5} />
            <ReferenceLine x={1 - eps} stroke="#E0304A" strokeDasharray="4 4" strokeOpacity={0.6} label={{ value: "1−ε", fontSize: 10, fill: "#FF6B6B", position: "top" }} />
            <ReferenceLine x={1 + eps} stroke="#E0304A" strokeDasharray="4 4" strokeOpacity={0.6} label={{ value: "1+ε", fontSize: 10, fill: "#FF6B6B", position: "top" }} />
            <Area dataKey="gap" name="被裁掉的区域" stroke="none" fill="#E0304A" fillOpacity={0.15} isAnimationActive={false} />
            <Line type="monotone" dataKey="cpi" name="无裁剪 r·A（旧目标）" stroke="#6B7280" strokeDasharray="5 4" dot={false} strokeWidth={1.5} isAnimationActive={false} />
            <Line type="monotone" dataKey="clip" name="PPO 裁剪目标" stroke="#F0B822" dot={false} strokeWidth={2.5} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="text-[13px] text-text-secondary leading-relaxed">
        {advSign === 1 ? (
          <>A &gt; 0：金色线在 r = 1+ε 处被「削平」——动作概率被推高超过 ε 之后，<b className="text-accent-yellow">目标不再增长</b>，PPO 失去继续猛推的动机；但 r &lt; 1−ε 的下滑侧<b className="text-accent-watermelon">不被保护</b>。红色阴影是"被裁掉的过度乐观区"。</>
        ) : (
          <>A &lt; 0：金色线在 r = 1−ε 处触底——概率被压低超过 ε 后，目标<b className="text-accent-yellow">不再下降</b>，PPO 不再继续狠压；而 r &gt; 1+ε 的上升侧照常生效。注意 min 取的是"更悲观"的那一支——这就是 PPO 的<b className="text-accent-yellow">不对称保护</b>：只拦"做过头"的方向。红色阴影是 r &lt; 1−ε 处被裁掉的"过度压价区"。</>
        )}
      </div>
    </div>
  );
}

/** Tab 2: GAE λ 光谱 —— 固定示例轨迹上的优势估计 */
const DELTAS = [0.5, -0.3, 0.8, 0.2, -0.6, 0.4, 0.9, 0.1, -0.2, 1.0];
const STEPS = DELTAS.map((_, i) => i);

function computeGAE(lam: number): number[] {
  const n = DELTAS.length;
  const A = new Array(n).fill(0);
  let acc = 0;
  for (let t = n - 1; t >= 0; t--) {
    acc = DELTAS[t] + GAMMA * lam * acc;
    A[t] = acc;
  }
  return A;
}

function GAEView() {
  const [lam, setLam] = useState(0.0);
  const A = useMemo(() => computeGAE(lam), [lam]);

  const data = STEPS.map(t => ({ step: t, adv: A[t], delta: DELTAS[t] }));

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <div className="flex-1">
          <div className="flex justify-between text-xs text-muted-foreground mb-1">
            <span>λ = {lam.toFixed(2)}（0 = TD(0) 低方差有偏 → 1 = MC 无偏高方差）</span>
          </div>
          <Slider value={[lam]} min={0} max={1} step={0.05} onValueChange={([l]) => setLam(l)} />
        </div>
      </div>

      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,92,246,0.15)" vertical={false} />
            <XAxis dataKey="step" stroke="#6B7280" fontSize={10} label={{ value: "步 t", position: "insideBottomRight", offset: -2, fontSize: 10, fill: "#6B7280" }} />
            <YAxis stroke="#6B7280" fontSize={10} />
            <Tooltip
              contentStyle={{ backgroundColor: "#0A0A0F", border: "1px solid rgba(139,92,246,0.3)", fontSize: 12 }}
              formatter={(v: number, name: string) => [v.toFixed(3), name]}
              labelFormatter={l => `t = ${l}`}
            />
            <ReferenceLine y={0} stroke="#6B7280" strokeOpacity={0.4} />
            <Bar dataKey="adv" name="优势 A_t^GAE" radius={[3, 3, 0, 0]}>
              {data.map((d) => (
                <Cell key={d.step} fill={d.adv >= 0 ? "#F0B822" : "#E0304A"} />
              ))}
            </Bar>
            <Line type="monotone" dataKey="delta" name="TD 误差 δ（灰虚线）" stroke="#9CA3AF" strokeDasharray="5 4" dot={{ r: 2, fill: "#9CA3AF" }} strokeWidth={1.5} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="text-[13px] text-text-secondary leading-relaxed">
        固定示例轨迹（灰虚线 δ 序列为 <span className="font-mono text-xs">[+.5, −.3, +.8, +.2, −.6, +.4, +.9, +.1, −.2, +1.0]</span>）。
        λ = 0 时 A_t 只看当前一步 δ（TD(0)）；λ 增大，每步"向后追溯"更多 TD 误差，A 的<b className="text-accent-yellow">幅度整体变大、序列变长</b>——
        λ = 1 时 A_t 退化为"该步之后全部真实 δ 之和"（MC 口径）。
        <b className="text-accent-yellow">GAE 的工程本质：用 λ 这个旋钮，在"MC 的无偏高方差"与"TD 的低方差有偏"之间连续插值。</b>
        代价函数左端的"序列长度"、右端的"噪声水平"，就是 Schulman 用一张超参图换来的全部权衡。
      </div>
    </div>
  );
}

export default function PPODemo() {
  const [tab, setTab] = useState("clip");
  return (
    <div className="rounded-xl border border-border bg-[#0A0A0F] shadow-glow-game overflow-hidden">
      <div className="px-4 py-3 border-b border-border/60 flex flex-wrap items-center justify-between gap-3">
        <div className="text-sm text-muted-foreground">
          <TeX math="L^{\mathrm{CLIP}}=\mathbb{E}\big[\min(r_t\hat{A}_t,\ \mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\hat{A}_t)\big]" />
        </div>
        <Tabs value={tab} onValueChange={setTab}>
          <TabsList>
            <TabsTrigger value="clip">① Clip 目标函数</TabsTrigger>
            <TabsTrigger value="gae">② GAE λ 光谱</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>
      <div className="p-4">
        {tab === "clip" ? <ClipView /> : <GAEView />}
      </div>
    </div>
  );
}
