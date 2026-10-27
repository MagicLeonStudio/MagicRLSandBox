import { useEffect, useRef } from "react";

export const ACTION_ARROWS = ["↑", "↓", "←", "→"];
export const ACTION_NAMES = ["上", "下", "左", "右"];

function hexToRgb(hex: string): [number, number, number] {
  const value = Number.parseInt(hex.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

export function PixelFrame({
  data,
  size = 48,
  className = "",
}: {
  data: Float32Array;
  size?: number;
  className?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const side = Math.sqrt(data.length / 3);
    for (let y = 0; y < side; y++) {
      for (let x = 0; x < side; x++) {
        const i = (y * side + x) * 3;
        const r = Math.round(Math.max(0, Math.min(1, data[i])) * 255);
        const g = Math.round(Math.max(0, Math.min(1, data[i + 1])) * 255);
        const b = Math.round(Math.max(0, Math.min(1, data[i + 2])) * 255);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(x, y, 1, 1);
      }
    }
  }, [data]);
  return (
    <canvas
      ref={ref}
      width={12}
      height={12}
      className={className}
      style={{ width: size, height: size, imageRendering: "pixelated" }}
    />
  );
}

function Heatmap({
  data,
  side,
  accent,
  size,
}: {
  data: Float32Array;
  side: number;
  accent: string;
  size: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const [r, g, b] = hexToRgb(accent);
    let max = 1e-6;
    for (const value of data) if (Number.isFinite(value)) max = Math.max(max, value);
    ctx.fillStyle = "#07060E";
    ctx.fillRect(0, 0, side, side);
    for (let y = 0; y < side; y++) {
      for (let x = 0; x < side; x++) {
        const raw = data[y * side + x];
        const value = Number.isFinite(raw) ? Math.max(0, raw) / max : 0;
        ctx.fillStyle = `rgba(${r},${g},${b},${0.06 + value * 0.94})`;
        ctx.fillRect(x, y, 1, 1);
      }
    }
  }, [data, side, accent]);
  return (
    <canvas
      ref={ref}
      width={side}
      height={side}
      className="rounded-[3px] border border-[#2A2445]"
      style={{ width: size, height: size, imageRendering: "pixelated" }}
    />
  );
}

export function FeatureGrid({
  data,
  side,
  filters,
  accent,
  size = 42,
}: {
  data: Float32Array;
  side: number;
  filters: number;
  accent: string;
  size?: number;
}) {
  const mapSize = side * side;
  return (
    <div className="flex flex-wrap gap-1.5">
      {Array.from({ length: filters }, (_, f) => (
        <Heatmap
          key={f}
          data={data.slice(f * mapSize, (f + 1) * mapSize)}
          side={side}
          accent={accent}
          size={size}
        />
      ))}
    </div>
  );
}

export function DenseBars({ data, accent }: { data: Float32Array; accent: string }) {
  let max = 1e-6;
  for (const value of data) if (Number.isFinite(value)) max = Math.max(max, value);
  return (
    <div className="flex h-12 items-end gap-[2px]" aria-label="Dense layer activations">
      {Array.from(data, (raw, i) => {
        const value = Number.isFinite(raw) ? Math.max(0, raw) / max : 0;
        return (
          <span
            key={i}
            className="w-[3px] rounded-sm"
            style={{ height: `${3 + value * 39}px`, backgroundColor: accent, opacity: 0.18 + value * 0.82 }}
          />
        );
      })}
    </div>
  );
}

export function QValueBars({
  q,
  accent,
  valid = [0, 1, 2, 3],
  compact = false,
}: {
  q: number[];
  accent: string;
  valid?: number[];
  compact?: boolean;
}) {
  const finite = q.map((value) => Number.isFinite(value) ? value : 0);
  const max = Math.max(0.1, ...finite.map((value) => Math.abs(value)));
  return (
    <div className={compact ? "space-y-1" : "space-y-1.5"}>
      {finite.map((value, action) => {
        const enabled = valid.includes(action);
        return (
          <div key={action} className="grid grid-cols-[20px_1fr_52px] items-center gap-2" style={{ opacity: enabled ? 1 : 0.25 }}>
            <span className="font-mono text-sm" style={{ color: accent }}>{ACTION_ARROWS[action]}</span>
            <div className="h-1.5 rounded-full bg-[#1C1832] overflow-hidden">
              <div
                className="h-full rounded-full"
                style={{ width: `${Math.abs(value) / max * 100}%`, backgroundColor: accent }}
              />
            </div>
            <span className="font-mono text-[11px] text-right text-[#A79FC7]">{value.toFixed(3)}</span>
          </div>
        );
      })}
    </div>
  );
}

export function Metric({
  label,
  value,
  accent = "#F1EEFB",
}: {
  label: string;
  value: string;
  accent?: string;
}) {
  return (
    <div className="min-w-0">
      <div className="text-[10px] uppercase tracking-[0.12em] text-[#6B6390]">{label}</div>
      <div className="truncate font-mono text-sm font-semibold" style={{ color: accent }}>{value}</div>
    </div>
  );
}
