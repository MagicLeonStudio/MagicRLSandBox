/**
 * ValueHeatmap —— GridWorld 价值/策略二维热图
 * 颜色语义（magic-theme）：深紫 = 低价值基底，金色 = 高价值（主角），红色仅用于风险标注。
 */
interface ValueHeatmapProps {
  size: number;
  values: number[];
  policy?: number[];        // 每格贪心动作（0上 1下 2左 3右，-1 无）
  walls?: Set<number> | number[];
  goal?: number;
  start?: number;
  showValues?: boolean;
  highlightPath?: number[]; // 收敛后最优路径高亮
  agentPos?: number;
  vmin?: number;            // 统一色域（多图对比时保证可比）
  vmax?: number;
}

const C_LOW = [42, 24, 69];    // #2A1845 深紫
const C_HIGH = [240, 184, 34]; // #F0B822 金

function lerpColor(t: number): string {
  const c = C_LOW.map((v, i) => Math.round(v + (C_HIGH[i] - v) * Math.min(1, Math.max(0, t))));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

const ARROWS = ["▲", "▼", "◀", "▶"]; // 0上 1下 2左 3右

export default function ValueHeatmap({
  size, values, policy, walls, goal, start = 0,
  showValues = true, highlightPath, agentPos, vmin, vmax,
}: ValueHeatmapProps) {
  const wallSet = walls instanceof Set ? walls : new Set(walls ?? []);
  const lo = vmin ?? Math.min(...values);
  const hi = vmax ?? Math.max(...values);
  const range = hi - lo || 1;
  const pathSet = new Set(highlightPath ?? []);

  return (
    <div
      className="grid gap-1 w-full max-w-[420px] select-none"
      style={{ gridTemplateColumns: `repeat(${size}, minmax(0,1fr))` }}
    >
      {values.map((v, s) => {
        if (wallSet.has(s)) {
          return (
            <div key={s} className="aspect-square rounded-md bg-[#15151f] border border-border/40 flex items-center justify-center">
              <span className="text-muted-foreground/40 text-xs">✕</span>
            </div>
          );
        }
        const isGoal = goal !== undefined && s === goal;
        const t = (v - lo) / range;
        const bg = isGoal ? "#F0B822" : lerpColor(t);
        const onPath = pathSet.has(s);
        return (
          <div
            key={s}
            className="aspect-square rounded-md flex flex-col items-center justify-center relative transition-colors duration-300"
            style={{
              backgroundColor: bg,
              boxShadow: onPath ? "0 0 0 2px #fff, 0 0 14px rgba(240,184,34,0.55)" : undefined,
            }}
          >
            {isGoal ? (
              <span className="text-[#2A1845] font-bold text-sm">★</span>
            ) : (
              <>
                {showValues && (
                  <span
                    className="text-[10px] md:text-xs font-mono leading-none"
                    style={{ color: t > 0.45 ? "#2A1845" : "rgba(240,240,255,0.85)" }}
                  >
                    {v.toFixed(1)}
                  </span>
                )}
                {policy && policy[s] >= 0 && (
                  <span
                    className="text-xs md:text-sm leading-none mt-0.5"
                    style={{ color: t > 0.45 ? "#2A1845" : "#F0B822", textShadow: t > 0.45 ? "none" : "0 0 6px rgba(240,184,34,0.6)" }}
                  >
                    {ARROWS[policy[s]]}
                  </span>
                )}
                {s === start && (
                  <span className="absolute top-0.5 left-1 text-[9px] font-bold" style={{ color: t > 0.45 ? "#2A1845" : "rgba(240,240,255,0.7)" }}>S</span>
                )}
              </>
            )}
            {agentPos === s && (
              <span className="absolute inset-0 flex items-center justify-center">
                <span className="w-3 h-3 rounded-full bg-white shadow-[0_0_10px_rgba(255,255,255,0.9)]" />
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}
