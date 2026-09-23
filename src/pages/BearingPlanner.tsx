import { Minus, TrendingDown, TrendingUp, UploadIcon } from "lucide-react";
import { useMemo, useState, type ChangeEvent } from "react";
import { Select } from "@/components/ui/select";
import {
  BEARING_TYPES,
  COUNTRIES,
  DRIVE_TYPES,
  MANUFACTURERS,
  MODELS,
  MOCK_SALES_PREDICTION,
  VEHICLE_MODELS,
} from "@/data/catalog";

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

export function BearingPlanner() {
  const [country, setCountry] = useState<string | null>(COUNTRIES[0].value);
  const [manufacturer, setManufacturer] = useState<string | null>(MANUFACTURERS[0].value);
  const [driveType, setDriveType] = useState<string | null>(DRIVE_TYPES[0].value);
  const [model, setModel] = useState<string | null>(MODELS[0].value);
  const [quantities, setQuantities] = useState<Record<string, string>>({});
  const [salesFileName, setSalesFileName] = useState<string | null>(null);
  const [salesFileError, setSalesFileError] = useState<string | null>(null);

  // ponytail: only one country/manufacturer/drive-type/model combination exists today,
  // so it always resolves to this single catalog entry.
  const vehicle = VEHICLE_MODELS[0];
  const prediction = MOCK_SALES_PREDICTION[vehicle.value];

  const entries = useMemo(
    () => BEARING_TYPES.map((b) => ({ ...b, qty: Number(quantities[b.value]) || 0 })),
    [quantities],
  );
  const total = entries.reduce((sum, e) => sum + e.qty, 0);

  function handleQtyChange(bearingValue: string, value: string) {
    setQuantities((prev) => ({ ...prev, [bearingValue]: value }));
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

  return (
    <main className="px-6 py-16">
      <section className="mx-auto max-w-[1000px] rounded-xl border border-faint-line bg-pure-white p-6 shadow-subtle">
        <h2 className="mb-1 text-[20px] font-semibold text-onyx">차량별 베어링 구성</h2>
        <p className="mb-6 text-[14px] text-warm-gray">
          판매국가·제조사·구동방식·차량모델을 선택하고, 베어링 4종 개수를 각각 입력하세요. 차량당 합계는 자동으로 계산됩니다.
        </p>

        <div className="overflow-x-auto">
          <table className="w-full min-w-[880px] border-collapse">
            <thead>
              <tr className="border-b border-faint-line text-left text-[12px] font-medium text-warm-gray">
                <th className="px-2 pb-2">판매국가</th>
                <th className="px-2 pb-2">제조사</th>
                <th className="px-2 pb-2">구동방식</th>
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
                      placeholder="0"
                      value={quantities[e.value] ?? ""}
                      onChange={(ev) => handleQtyChange(e.value, ev.target.value)}
                      className="h-11 w-20 rounded-lg border border-faint-line bg-pure-white px-3 text-[16px] text-onyx outline-none transition-colors focus:border-signal-blue focus:ring-2 focus:ring-signal-blue/25"
                    />
                  </td>
                ))}
                <td className="px-2 py-3 align-top">
                  <div className="flex h-11 min-w-20 items-center rounded-lg bg-paper-white px-3 text-[16px] font-semibold text-onyx">
                    {total.toLocaleString()}개
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
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
                  {e.label} <span className="text-warm-gray">· {e.qty}개 / 대</span>
                </span>
                <span className="text-[15px] font-semibold text-onyx">
                  {(prediction.currentMonth * e.qty).toLocaleString()}개
                </span>
              </div>
            ))}
            <div className="flex items-center justify-between border-t border-faint-line bg-paper-white px-4 py-3">
              <span className="text-[15px] font-semibold text-onyx">합계</span>
              <span className="text-[15px] font-semibold text-onyx">
                {(prediction.currentMonth * total).toLocaleString()}개
              </span>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
