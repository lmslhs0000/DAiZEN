import { ArrowLeft } from "lucide-react";
import { SalesTrendChart } from "@/components/SalesTrendChart";
import { MOCK_MONTHLY_SALES, VEHICLE_MODELS } from "@/data/catalog";
import { taskTitle, type DemandForecastTask } from "@/data/tasks";

interface TaskChartProps {
  task: DemandForecastTask;
  onBack: () => void;
}

export function TaskChart({ task, onBack }: TaskChartProps) {
  // ponytail: only one vehicle exists in the mock catalog today, so every task resolves to it.
  const vehicle = VEHICLE_MODELS[0];
  const series = MOCK_MONTHLY_SALES[vehicle.value] ?? [];
  const latest = series.at(-1)?.sales ?? 0;

  return (
    <main className="px-6 py-16">
      <button
        type="button"
        onClick={onBack}
        className="mb-6 flex items-center gap-1 text-[14px] font-medium text-signal-blue hover:underline"
      >
        <ArrowLeft className="h-4 w-4" aria-hidden="true" />
        작업 목록으로
      </button>

      <div className="mb-2 flex items-center gap-2">
        <h2 className="text-[20px] font-semibold text-onyx">{taskTitle(task)} — 예상 판매량 추이</h2>
        <span className="rounded-full bg-paper-white px-3 py-1 text-[12px] font-medium text-warm-gray">
          샘플 데이터 · 백엔드 연동 전
        </span>
      </div>

      <div className="mt-8 max-w-[480px] rounded-xl border border-faint-line bg-pure-white p-4 shadow-subtle">
        <div className="mb-3 flex items-center justify-between">
          <p className="text-[15px] font-semibold text-onyx">{vehicle.label}</p>
          <p className="text-[13px] text-warm-gray">
            이번 달 <span className="font-semibold text-onyx">{latest.toLocaleString()}대</span>
          </p>
        </div>
        <SalesTrendChart data={series} />
      </div>
    </main>
  );
}
