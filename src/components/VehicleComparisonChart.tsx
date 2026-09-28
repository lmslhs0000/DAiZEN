import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ReactNode } from "react";

export interface ComparisonSeries {
  id: string;
  label: string;
  color: string;
  points: { month: string; value: number }[];
}

export type ChartMode = "actual" | "indexed";

interface VehicleComparisonChartProps {
  series: ComparisonSeries[];
  mode: ChartMode;
  height?: number;
}

function formatValue(value: number) {
  return Math.round(value).toLocaleString();
}

// month is "YYYY-MM" so multi-year series (up to 24 months) never repeat an ambiguous label.
function formatMonthTick(value: string) {
  const [y, m] = value.split("-");
  return y && m ? `${y.slice(2)}.${m}` : value;
}

function formatMonthLabel(value: ReactNode): ReactNode {
  if (typeof value !== "string") return value;
  const [y, m] = value.split("-");
  return y && m ? `${y}년 ${Number(m)}월` : value;
}

// 추세 비교: 각 시리즈의 첫 값을 100으로 환산해, 절대 규모가 달라도 변화 속도를 비교할 수 있게 한다.
function toIndexedSeries(series: ComparisonSeries[]): ComparisonSeries[] {
  return series.map((s) => {
    const base = s.points[0]?.value ?? 0;
    return {
      ...s,
      points: s.points.map((p) => ({ month: p.month, value: base > 0 ? (p.value / base) * 100 : 0 })),
    };
  });
}

// 여러 시리즈를 월(month) 기준으로 하나의 recharts 데이터 배열로 합친다 — 시리즈마다 dataKey가 다른 필드로 들어간다.
function buildComparisonData(series: ComparisonSeries[]) {
  const months = Array.from(new Set(series.flatMap((s) => s.points.map((p) => p.month)))).sort();
  return months.map((month) => {
    const row: Record<string, string | number> = { month };
    for (const s of series) {
      const point = s.points.find((p) => p.month === month);
      if (point) row[s.id] = point.value;
    }
    return row;
  });
}

export function VehicleComparisonChart({ series, mode, height = 320 }: VehicleComparisonChartProps) {
  const isIndexed = mode === "indexed";
  const displaySeries = isIndexed ? toIndexedSeries(series) : series;
  const data = buildComparisonData(displaySeries);
  const maxTicks = 6;
  const tickInterval = data.length > maxTicks ? Math.ceil(data.length / maxTicks) - 1 : 0;

  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 24, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="#dddddd" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="month"
            tickLine={false}
            axisLine={false}
            tick={{ fill: "#615d59", fontSize: 12 }}
            tickFormatter={formatMonthTick}
            interval={tickInterval}
          />
          <YAxis
            tickLine={false}
            axisLine={false}
            tick={{ fill: "#615d59", fontSize: 12 }}
            tickFormatter={isIndexed ? (v: number) => `${v}` : formatValue}
            width={56}
            domain={isIndexed ? [0, 400] : ["auto", "auto"]}
            ticks={isIndexed ? [0, 100, 200, 300, 400] : undefined}
          />
          {isIndexed && <ReferenceLine y={100} stroke="#a39e98" strokeDasharray="4 4" />}
          <Tooltip
            labelFormatter={formatMonthLabel}
            formatter={(value, name) => [
              isIndexed ? formatValue(Number(value)) : `${formatValue(Number(value))}개`,
              name,
            ]}
            contentStyle={{
              borderRadius: 8,
              borderColor: "#dddddd",
              fontSize: 13,
            }}
          />
          {displaySeries.map((s) => (
            <Line
              key={s.id}
              type="monotone"
              dataKey={s.id}
              name={s.label}
              stroke={s.color}
              strokeWidth={2}
              dot={{ r: 3, fill: s.color }}
              activeDot={{ r: 6 }}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
