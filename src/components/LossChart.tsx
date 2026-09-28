import {
  AreaChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface LossChartProps {
  data: number[];
}

function computeMovingAverage(data: number[], window: number): number[] {
  const ma: number[] = [];
  for (let i = 0; i < data.length; i++) {
    const start = Math.max(0, i - window + 1);
    const slice = data.slice(start, i + 1);
    const avg = slice.reduce((a, b) => b + a, 0) / slice.length;
    ma.push(avg);
  }
  return ma;
}

export default function LossChart({ data }: LossChartProps) {
  const maData = computeMovingAverage(data, 20);

  const chartData = data.map((value, index) => ({
    episode: index + 1,
    loss: value,
    ma: maData[index],
  }));

  return (
    <div className="bg-bg-elevated rounded-lg p-4 border border-[rgba(139,92,246,0.08)]">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-text-primary text-[13px] font-semibold">Loss</h3>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span
              className="inline-block w-4 h-px"
              style={{ backgroundColor: "#FF6B6B", opacity: 0.3 }}
            />
            <span className="text-text-muted text-[10px]">Raw</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span
              className="inline-block w-4 h-0.5"
              style={{ backgroundColor: "#FF6B6B", opacity: 1 }}
            />
            <span className="text-text-muted text-[10px]">MA(20)</span>
          </div>
        </div>
      </div>
      <div style={{ height: 140 }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="lossGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#FF6B6B" stopOpacity={0.1} />
                <stop offset="95%" stopColor="#FF6B6B" stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="episode"
              tick={{ fill: "#6B7280", fontSize: 10 }}
              tickLine={false}
              axisLine={false}
              interval="preserveStartEnd"
              minTickGap={30}
            />
            <YAxis
              tick={{ fill: "#6B7280", fontSize: 10 }}
              tickLine={false}
              axisLine={false}
              width={40}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#12121A",
                border: "1px solid rgba(139, 92, 246, 0.2)",
                borderRadius: 4,
                padding: "4px 8px",
                fontSize: 12,
              }}
              labelStyle={{ color: "#9CA3AF" }}
              formatter={(value: number, name: string) => {
                if (name === "ma") return [value.toFixed(4), "MA(20)"];
                return [value.toFixed(4), "Loss"];
              }}
              labelFormatter={(label: number) => `EP ${label}`}
            />
            {/* Raw data: thin, semi-transparent line */}
            <Line
              type="monotone"
              dataKey="loss"
              stroke="#FF6B6B"
              strokeWidth={1}
              strokeOpacity={0.3}
              dot={false}
              isAnimationActive={false}
            />
            {/* Moving average: thicker, full opacity line with area fill */}
            <Area
              type="monotone"
              dataKey="ma"
              stroke="#FF6B6B"
              strokeWidth={2}
              strokeOpacity={1}
              fill="url(#lossGradient)"
              isAnimationActive={false}
              dot={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
