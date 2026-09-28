import { ArrowLeft } from "lucide-react";
import { useMemo, useState } from "react";
import { VehicleComparisonChart, type ChartMode, type ComparisonSeries } from "@/components/VehicleComparisonChart";
import { findVehicle, MOCK_MONTHLY_SALES } from "@/data/catalog";
import {
  computeBearingTotal,
  resolveVehicleStatus,
  vehicleComboLabel,
  type DemandForecastTask,
} from "@/data/tasks";

interface TaskChartProps {
  tasks: DemandForecastTask[];
  onBack: () => void;
}

// 계열이 늘어나도 구분되도록 순환하는 카테고리 팔레트 — 브랜드 파란색과 겹치지 않는 값들로 구성.
const SERIES_COLORS = ["#0075de", "#f2994a", "#27ae60", "#9b51e0", "#eb5757", "#2f80ed"];

export function TaskChart({ tasks, onBack }: TaskChartProps) {
  // 베어링 생산사는 국가와 무관하게 전체 차량 수요를 한눈에 봐야 하므로, 국가로 나누지 않고
  // 차량 마스터가 확정된(RESOLVED) 작업 전체를 비교 대상으로 삼는다.
  const comparableTasks = useMemo(
    () => tasks.filter((t) => resolveVehicleStatus(t) === "RESOLVED"),
    [tasks],
  );

  const [checkedIds, setCheckedIds] = useState<Set<string>>(
    () => new Set(comparableTasks.map((t) => t.id)),
  );
  const [mode, setMode] = useState<ChartMode>("actual");

  function toggleTask(id: string, checked: boolean) {
    setCheckedIds((prev) => {
      const next = new Set(prev);
      if (checked) next.add(id);
      else next.delete(id);
      return next;
    });
  }

  const allSeries: (ComparisonSeries & { taskId: string })[] = comparableTasks.map((t, i) => {
    const vehicle = findVehicle(t)!;
    const bearingTotal = computeBearingTotal(t);
    const sales = MOCK_MONTHLY_SALES[vehicle.value] ?? [];
    return {
      id: t.id,
      taskId: t.id,
      label: vehicleComboLabel(t),
      color: SERIES_COLORS[i % SERIES_COLORS.length],
      points: sales.map((s) => ({ month: s.month, value: s.sales * bearingTotal })),
    };
  });

  const visibleSeries = allSeries.filter((s) => checkedIds.has(s.taskId));

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

      <div className="mb-2 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <h2 className="text-[20px] font-semibold text-onyx">차량별 수요 추세</h2>
          <span className="rounded-full bg-paper-white px-3 py-1 text-[12px] font-medium text-warm-gray">
            샘플 데이터 · 백엔드 연동 전
          </span>
        </div>
        <div className="flex gap-1 rounded-lg border border-faint-line bg-paper-white p-1">
          <button
            type="button"
            onClick={() => setMode("indexed")}
            className={`rounded-md px-3 py-1.5 text-[13px] font-medium transition-colors ${
              mode === "indexed" ? "bg-midnight-ink text-pure-white" : "text-warm-gray hover:text-onyx"
            }`}
          >
            추세 비교
          </button>
          <button
            type="button"
            onClick={() => setMode("actual")}
            className={`rounded-md px-3 py-1.5 text-[13px] font-medium transition-colors ${
              mode === "actual" ? "bg-midnight-ink text-pure-white" : "text-warm-gray hover:text-onyx"
            }`}
          >
            실제 수요량
          </button>
        </div>
      </div>
      <p className="mb-6 text-[14px] text-warm-gray">
        {mode === "indexed"
          ? "추세 비교는 각 차량의 첫 예측을 100으로 환산해 변화 속도를 비교합니다."
          : "저장된 작업들을 체크박스로 켜고 꺼서 베어링 예상 수요를 비교할 수 있습니다."}
      </p>

      {comparableTasks.length === 0 ? (
        <div className="max-w-[720px] rounded-xl border border-faint-line bg-pure-white p-8 text-center text-[14px] text-warm-gray">
          차량이 확정된 작업이 없습니다.
        </div>
      ) : (
        <div className="max-w-[1200px] rounded-xl border border-faint-line bg-pure-white p-6 shadow-subtle">
          <div className="mb-4 flex flex-wrap gap-x-6 gap-y-2">
            {allSeries.map((s) => (
              <label key={s.taskId} className="flex items-center gap-2 text-[13px] text-onyx">
                <input
                  type="checkbox"
                  checked={checkedIds.has(s.taskId)}
                  onChange={(e) => toggleTask(s.taskId, e.target.checked)}
                  className="h-3.5 w-3.5 accent-signal-blue"
                />
                <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: s.color }} aria-hidden="true" />
                {s.label}
              </label>
            ))}
          </div>

          <p className="mb-2 text-[12px] text-warm-gray">
            {mode === "indexed" ? "첫 예측 대비 지수" : "베어링 예상 수요 (개)"}
          </p>
          {visibleSeries.length > 0 ? (
            <VehicleComparisonChart series={visibleSeries} mode={mode} height={440} />
          ) : (
            <div className="flex h-[440px] items-center justify-center text-[14px] text-warm-gray">
              비교할 차량을 체크박스에서 선택하세요.
            </div>
          )}
        </div>
      )}
    </main>
  );
}
