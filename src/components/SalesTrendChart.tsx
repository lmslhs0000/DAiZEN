import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ReactNode } from "react";

interface SalesTrendChartProps {
  data: { month: string; sales: number }[];
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

export function SalesTrendChart({ data, height = 200 }: SalesTrendChartProps) {
  // Thin out ticks as the series grows so labels never crowd in a narrow card, whether it's 6 months or 24.
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
            tickFormatter={formatValue}
            width={56}
          />
          <Tooltip
            cursor={{ stroke: "#0075de", strokeDasharray: "4 4" }}
            labelFormatter={formatMonthLabel}
            formatter={(value) => [`${formatValue(Number(value))}대`, "예상 판매량"]}
            contentStyle={{
              borderRadius: 8,
              borderColor: "#dddddd",
              fontSize: 13,
            }}
          />
          <Line
            type="monotone"
            dataKey="sales"
            stroke="#0075de"
            strokeWidth={2}
            dot={{ r: 3, fill: "#0075de" }}
            activeDot={{ r: 6 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
