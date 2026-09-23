import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

interface SalesTrendChartProps {
  data: { month: string; sales: number }[];
  height?: number;
}

function formatValue(value: number) {
  return Math.round(value).toLocaleString();
}

export function SalesTrendChart({ data, height = 200 }: SalesTrendChartProps) {
  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="#dddddd" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="month"
            tickLine={false}
            axisLine={false}
            tick={{ fill: "#615d59", fontSize: 12 }}
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
