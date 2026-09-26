import { ArrowLeft, Minus, TrendingDown, TrendingUp, UploadIcon } from "lucide-react";
import { useMemo, useState, type ChangeEvent } from "react";
import { Select } from "@/components/ui/select";
import { VehicleStatusBadge } from "@/components/VehicleStatusBadge";
import {
  BEARING_TYPES,
  COUNTRIES,
  DRIVE_TYPES,
  MANUFACTURERS,
  MODELS,
  MOCK_SALES_PREDICTION,
  VEHICLE_MODELS,
} from "@/data/catalog";
import { formatTaskTimestamp, resolveVehicleStatus, type DemandForecastTask } from "@/data/tasks";

function DeltaBadge({ label, pct }: { label: string; pct: number }) {
  const isFlat = Math.abs(pct) < 0.05;
  const isUp = pct > 0;
  const Icon = isFlat ? Minus : isUp ? TrendingUp : TrendingDown;
  const color = isFlat ? "text-warm-gray" : isUp ? "text-signal-blue" : "text-decrease";

  return (
    <div className="rounded-xl border border-faint-line bg-pure-white p-4">
      <p className="text-[13px] text-warm-gray">{label}</p>
      <div className={`mt-1 flex items-center gap-1 text-[20px] font-semibold ${color}`}>
        <Icon className="h-4 w-4" aria-hidden="true" />
        <span>
          {isUp ? "+" : ""}
          {pct.toFixed(1)}%
        </span>
      </div>
    </div>
  );
}

interface TaskDetailProps {
  task: DemandForecastTask;
  onSave: (patch: Partial<DemandForecastTask>) => void;
  onBack: () => void;
}

export function TaskDetail({ task, onSave, onBack }: TaskDetailProps) {
  const [country, setCountry] = useState<string | null>(task.country);
  const [manufacturer, setManufacturer] = useState<string | null>(task.manufacturer);
  const [driveType, setDriveType] = useState<string | null>(task.driveType);
  const [model, setModel] = useState<string | null>(task.model);
  const [quantities, setQuantities] = useState<Record<string, string>>(task.quantities);
  const [naFlags, setNaFlags] = useState<Record<string, boolean>>(task.naFlags);
  const [salesFileName, setSalesFileName] = useState<string | null>(null);
  const [salesFileError, setSalesFileError] = useState<string | null>(null);

  // ponytail: only one country/manufacturer/drive-type/model combination exists today,
  // so it always resolves to this single catalog entry.
  const vehicle = VEHICLE_MODELS[0];
  const prediction = MOCK_SALES_PREDICTION[vehicle.value];

  // 베어링 입력은 실제 0 / N/A(해당없음) / 미설정을 구분한다 — 빈 값을 0으로 취급하지 않는다.
  const entries = useMemo(
    () =>
      BEARING_TYPES.map((b) => {
        const isNA = naFlags[b.value] ?? false;
        const raw = quantities[b.value] ?? "";
        const status: "value" | "na" | "unset" = isNA ? "na" : raw === "" ? "unset" : "value";
        return { ...b, status, qty: status === "value" ? Number(raw) : 0 };
      }),
    [quantities, naFlags],
  );
  const definedEntries = entries.filter((e) => e.status === "value");
  const hasDefined = definedEntries.length > 0;
  const total = definedEntries.reduce((sum, e) => sum + e.qty, 0);
  const vehicleStatus = resolveVehicleStatus({ country, manufacturer, driveType, model });

  function handleQtyChange(bearingValue: string, value: string) {
    setQuantities((prev) => ({ ...prev, [bearingValue]: value }));
  }

  function handleNaChange(bearingValue: string, checked: boolean) {
    setNaFlags((prev) => ({ ...prev, [bearingValue]: checked }));
    if (checked) {
      setQuantities((prev) => ({ ...prev, [bearingValue]: "" }));
    }
  }

  function handleSalesFileChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) {
      setSalesFileName(null);
      setSalesFileError(null);
      return;
    }
    if (!/\.(csv|xlsx)$/i.test(file.name)) {
      setSalesFileError("CSV 또는 XLSX 파일만 업로드할 수 있습니다.");
      setSalesFileName(null);
      return;
    }
    setSalesFileError(null);
    setSalesFileName(file.name);
  }

  function handleSave() {
    // 차량/베어링 입력이 완성되지 않아도 그대로 저장한다 — 검증 없이 현재 상태를 그대로 반영.
    onSave({ country, manufacturer, driveType, model, quantities, naFlags });
  }

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

      <section className="mx-auto max-w-[1000px] rounded-xl border border-faint-line bg-pure-white p-6 shadow-subtle">
        <div className="mb-1 flex items-center gap-2">
          <h2 className="text-[20px] font-semibold text-onyx">차량별 베어링 구성</h2>
          <VehicleStatusBadge status={vehicleStatus} />
        </div>
        <p className="mb-6 text-[14px] text-warm-gray">
          판매국가·구동방식·제조사·차량모델을 선택하고, 베어링 4종 개수를 각각 입력하세요. 차량당 합계는 자동으로 계산됩니다.
        </p>

        <div className="overflow-x-auto">
          <table className="w-full min-w-[880px] border-collapse">
            <thead>
              <tr className="border-b border-faint-line text-left text-[12px] font-medium text-warm-gray">
                <th className="px-2 pb-2">판매국가</th>
                <th className="px-2 pb-2">구동방식</th>
                <th className="px-2 pb-2">제조사</th>
                <th className="px-2 pb-2">차량모델</th>
                {BEARING_TYPES.map((b) => (
                  <th key={b.value} className="px-2 pb-2">
                    {b.label}
                  </th>
                ))}
                <th className="px-2 pb-2">차량당 합계</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="min-w-[110px] px-2 py-3 align-top">
                  <Select
                    hideLabel
                    label="판매국가"
                    placeholder="선택"
                    options={COUNTRIES}
                    value={country}
                    onChange={setCountry}
                  />
                </td>
                <td className="min-w-[110px] px-2 py-3 align-top">
                  <Select
                    hideLabel
                    label="구동방식"
                    placeholder="선택"
                    options={DRIVE_TYPES}
                    value={driveType}
                    onChange={setDriveType}
                  />
                </td>
                <td className="min-w-[110px] px-2 py-3 align-top">
                  <Select
                    hideLabel
                    label="제조사"
                    placeholder="선택"
                    options={MANUFACTURERS}
                    value={manufacturer}
                    onChange={setManufacturer}
                  />
                </td>
                <td className="min-w-[110px] px-2 py-3 align-top">
                  <Select
                    hideLabel
                    label="차량모델"
                    placeholder="선택"
                    options={MODELS}
                    value={model}
                    onChange={setModel}
                  />
                </td>
                {entries.map((e) => (
                  <td key={e.value} className="px-2 py-3 align-top">
                    <label htmlFor={`qty-${e.value}`} className="sr-only">
                      {e.label}
                    </label>
                    <input
                      id={`qty-${e.value}`}
                      type="number"
                      min={0}
                      step={1}
                      inputMode="numeric"
                      placeholder="미설정"
                      disabled={e.status === "na"}
                      value={quantities[e.value] ?? ""}
                      onChange={(ev) => handleQtyChange(e.value, ev.target.value)}
                      className="h-11 w-24 rounded-lg border border-faint-line bg-pure-white px-3 text-[16px] text-onyx outline-none transition-colors focus:border-signal-blue focus:ring-2 focus:ring-signal-blue/25 disabled:bg-paper-white disabled:text-warm-gray"
                    />
                    <label className="mt-1.5 flex items-center gap-1.5 text-[12px] text-warm-gray">
                      <input
                        type="checkbox"
                        checked={e.status === "na"}
                        onChange={(ev) => handleNaChange(e.value, ev.target.checked)}
                        className="h-3.5 w-3.5 accent-signal-blue"
                      />
                      해당없음
                    </label>
                  </td>
                ))}
                <td className="px-2 py-3 align-top">
                  <div className="flex h-11 min-w-20 items-center rounded-lg bg-paper-white px-3 text-[16px] font-semibold text-onyx">
                    {hasDefined ? `${total.toLocaleString()}개` : "-"}
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-[12px] text-warm-gray">미설정 항목은 차량당 합계에서 제외됩니다.</p>
      </section>

      <section className="mx-auto mt-6 max-w-[1000px] rounded-xl border border-faint-line bg-pure-white p-6 shadow-subtle">
        <div className="flex items-center justify-between gap-4">
          <div>
            <p className="text-[14px] font-medium text-onyx">현재 작업 저장</p>
            <p className="mt-1 text-[13px] text-warm-gray">
              차량·베어링 입력이 완성되지 않아도 저장할 수 있습니다.
            </p>
          </div>
          <button
            type="button"
            onClick={handleSave}
            className="h-11 shrink-0 rounded-lg bg-signal-blue px-5 text-[15px] font-medium text-pure-white transition-opacity hover:opacity-90"
          >
            저장
          </button>
        </div>
        <p className="mt-2 text-[12px] text-warm-gray">최근 저장 · {formatTaskTimestamp(task.updatedAt)}</p>
      </section>

      <section className="mx-auto mt-6 max-w-[1000px] rounded-xl border border-faint-line bg-pure-white p-6 shadow-subtle">
        <label htmlFor="sales-file" className="mb-2 flex items-center gap-1.5 text-[14px] font-medium text-onyx">
          <UploadIcon className="h-4 w-4 opacity-60" aria-hidden="true" />
          판매량 데이터 업로드
        </label>
        <input
          id="sales-file"
          type="file"
          accept=".csv,.xlsx"
          onChange={handleSalesFileChange}
          className="block w-full text-[14px] text-onyx file:mr-4 file:rounded-lg file:border file:border-faint-line file:bg-paper-white file:px-4 file:py-2 file:text-[13px] file:font-medium file:text-onyx hover:file:bg-faint-line/30"
        />
        <p className="mt-2 text-[13px] text-warm-gray">
          CSV 또는 XLSX 형식의 월별 판매량 데이터를 업로드하세요.
        </p>
        {salesFileError ? (
          <p className="mt-1 text-[13px] text-decrease">{salesFileError}</p>
        ) : (
          salesFileName && <p className="mt-1 text-[13px] text-warm-gray">선택된 파일: {salesFileName}</p>
        )}
      </section>

      <section className="mx-auto mt-16 max-w-[720px]">
        <div className="mb-4 flex items-center gap-2">
          <h2 className="text-[20px] font-semibold text-onyx">{vehicle.label} — 예측 결과</h2>
          <span className="rounded-full bg-paper-white px-3 py-1 text-[12px] font-medium text-warm-gray">
            샘플 데이터 · 백엔드 연동 전
          </span>
        </div>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <DeltaBadge
            label="전월 대비"
            pct={((prediction.currentMonth - prediction.prevMonth) / prediction.prevMonth) * 100}
          />
          <DeltaBadge
            label="반기 평균 대비"
            pct={((prediction.currentMonth - prediction.halfYearAvg) / prediction.halfYearAvg) * 100}
          />
          <DeltaBadge
            label="전년 동월 대비"
            pct={((prediction.currentMonth - prediction.prevYear) / prediction.prevYear) * 100}
          />
        </div>

        <p className="mt-4 text-[14px] text-warm-gray">
          이번 달 예상 판매량{" "}
          <span className="font-semibold text-onyx">{prediction.currentMonth.toLocaleString()}대</span>
        </p>

        <div className="mt-6">
          <h3 className="mb-3 text-[15px] font-semibold text-onyx">베어링별 생산 필요량</h3>
          <div className="overflow-hidden rounded-lg border border-faint-line">
            {entries.map((e, i) => (
              <div
                key={e.value}
                className={`flex items-center justify-between bg-pure-white px-4 py-3 ${i > 0 ? "border-t border-faint-line" : ""}`}
              >
                <span className="text-[15px] text-onyx">
                  {e.label}{" "}
                  <span className="text-warm-gray">
                    · {e.status === "value" ? `${e.qty}개 / 대` : e.status === "na" ? "해당없음" : "미설정"}
                  </span>
                </span>
                <span className="text-[15px] font-semibold text-onyx">
                  {e.status === "value" ? `${(prediction.currentMonth * e.qty).toLocaleString()}개` : "-"}
                </span>
              </div>
            ))}
            <div className="flex items-center justify-between border-t border-faint-line bg-paper-white px-4 py-3">
              <span className="text-[15px] font-semibold text-onyx">합계</span>
              <span className="text-[15px] font-semibold text-onyx">
                {hasDefined ? `${(prediction.currentMonth * total).toLocaleString()}개` : "-"}
              </span>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
