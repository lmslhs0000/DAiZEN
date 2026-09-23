import { SalesTrendChart } from "@/components/SalesTrendChart";
import { MOCK_MONTHLY_SALES, VEHICLE_MODELS } from "@/data/catalog";

export function ChartPage() {
  return (
    <main className="px-6 py-16">
      <div className="mb-2 flex items-center gap-2">
        <h2 className="text-[20px] font-semibold text-onyx">차종별 예상 판매량 추이</h2>
        <span className="rounded-full bg-paper-white px-3 py-1 text-[12px] font-medium text-warm-gray">
          샘플 데이터 · 백엔드 연동 전
        </span>
      </div>

      <div className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-2">
        {VEHICLE_MODELS.map((v) => {
          const series = MOCK_MONTHLY_SALES[v.value] ?? [];
          const latest = series.at(-1)?.sales ?? 0;
          return (
            <div key={v.value} className="rounded-xl border border-faint-line bg-pure-white p-4 shadow-subtle">
              <div className="mb-3 flex items-center justify-between">
                <p className="text-[15px] font-semibold text-onyx">{v.label}</p>
                <p className="text-[13px] text-warm-gray">
                  이번 달 <span className="font-semibold text-onyx">{latest.toLocaleString()}대</span>
                </p>
              </div>
              <SalesTrendChart data={series} />
            </div>
          );
        })}
      </div>
    </main>
  );
}
