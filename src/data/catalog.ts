export interface VehicleModel {
  value: string;
  label: string;
  country: string;
  manufacturer: string;
  driveType: string;
  model: string;
}

/**
 * ponytail: full vehicle catalog — every real (country, manufacturer, driveType, model)
 * combination the mock data knows about. Backed by real API data later.
 */
export const VEHICLE_MODELS: VehicleModel[] = [
  { value: "tesla-cybercab", label: "Tesla Cybercab", country: "kr", manufacturer: "tesla", driveType: "ev", model: "cybercab" },
  { value: "honda-cr-v", label: "Honda CR-V", country: "ca", manufacturer: "honda", driveType: "ice", model: "cr-v" },
  { value: "kia-ev6", label: "Kia EV6", country: "ca", manufacturer: "kia", driveType: "ev", model: "ev6" },
  { value: "kia-niro", label: "Kia Niro", country: "ca", manufacturer: "kia", driveType: "ev", model: "niro" },
  { value: "hyundai-porter-ii", label: "Hyundai Porter Ⅱ", country: "kr", manufacturer: "hyundai", driveType: "ev", model: "porter-ii" },
];

/** Finds the catalog vehicle matching a (country, manufacturer, driveType, model) selection, if any. */
export function findVehicle(selection: {
  country: string | null;
  manufacturer: string | null;
  driveType: string | null;
  model: string | null;
}): VehicleModel | undefined {
  return VEHICLE_MODELS.find(
    (v) =>
      v.country === selection.country &&
      v.manufacturer === selection.manufacturer &&
      v.driveType === selection.driveType &&
      v.model === selection.model,
  );
}

/**
 * 선택된 제조사·구동방식에 실제로 존재하는 차량모델만 반환한다 (예: Tesla → Cybercab만,
 * Honda → CR-V만, Kia → EV6·Niro, Hyundai → Porter Ⅱ만). 구동방식까지 정해지면 그 안에서도
 * 다시 좁힌다 (CR-V는 ICE 전용이라 구동방식을 EV로 바꾸면 목록에서 사라진다).
 */
export function getAvailableModels(
  manufacturer: string | null,
  driveType: string | null,
): { label: string; value: string }[] {
  const matches = VEHICLE_MODELS.filter(
    (v) =>
      (manufacturer === null || v.manufacturer === manufacturer) &&
      (driveType === null || v.driveType === driveType),
  );
  const seen = new Set<string>();
  const options: { label: string; value: string }[] = [];
  for (const v of matches) {
    if (seen.has(v.model)) continue;
    seen.add(v.model);
    options.push({ label: MODELS.find((m) => m.value === v.model)?.label ?? v.model, value: v.model });
  }
  return options;
}

/**
 * ponytail: option lists for the vehicle-config table row selects, to be replaced by real API
 * data. "ROBOTAXI" is a vehicle *role*, managed separately by the backend — it is not a model
 * name, so it must not be hardcoded here as one.
 */
export const COUNTRIES = [
  { label: "한국", value: "kr" },
  { label: "Canada", value: "ca" },
];
export const MANUFACTURERS = [
  { label: "Tesla", value: "tesla" },
  { label: "Honda", value: "honda" },
  { label: "Kia", value: "kia" },
  { label: "Hyundai", value: "hyundai" },
];
export const DRIVE_TYPES = [
  { label: "EV", value: "ev" },
  { label: "ICE", value: "ice" },
];
export const MODELS = [
  { label: "Cybercab", value: "cybercab" },
  { label: "CR-V", value: "cr-v" },
  { label: "EV6", value: "ev6" },
  { label: "Niro", value: "niro" },
  { label: "Porter Ⅱ", value: "porter-ii" },
];

export const BEARING_TYPES = [
  { label: "ENGINE", value: "engine" },
  { label: "TRANSMISSION", value: "transmission" },
  { label: "REDUCER", value: "reducer" },
  { label: "STEERING", value: "steering" },
];

/**
 * ponytail: deterministic mock generator instead of hand-typed arrays — produces a 24-month
 * forward-looking series (current month + up to 24 months ahead) per vehicle, with a steady
 * growth trend plus a mild seasonal wave so the line isn't a flat ramp.
 */
function buildMonthlySeries(
  startYear: number,
  startMonth: number,
  months: number,
  base: number,
  monthlyGrowth: number,
  waveAmplitude: number,
): { month: string; sales: number }[] {
  const series: { month: string; sales: number }[] = [];
  let y = startYear;
  let m = startMonth;
  for (let i = 0; i < months; i++) {
    const trend = base * Math.pow(1 + monthlyGrowth, i);
    const wave = Math.sin(i / 3) * waveAmplitude;
    series.push({ month: `${y}-${String(m).padStart(2, "0")}`, sales: Math.max(0, Math.round(trend + wave)) });
    m += 1;
    if (m > 12) {
      m = 1;
      y += 1;
    }
  }
  return series;
}

const FORECAST_MONTHS = 24;
const FORECAST_START: [year: number, month: number] = [2025, 4];

/**
 * `month` is "YYYY-MM" and each series runs 24 months forward from {@link FORECAST_START} —
 * the current month plus up to 24 months of future bearing demand.
 */
export const MOCK_MONTHLY_SALES: Record<string, { month: string; sales: number }[]> = {
  "tesla-cybercab": buildMonthlySeries(...FORECAST_START, FORECAST_MONTHS, 10500, 0.012, 400),
  "honda-cr-v": buildMonthlySeries(...FORECAST_START, FORECAST_MONTHS, 3200, 0.004, 150),
  "kia-ev6": buildMonthlySeries(...FORECAST_START, FORECAST_MONTHS, 950, 0.012, 80),
  "kia-niro": buildMonthlySeries(...FORECAST_START, FORECAST_MONTHS, 620, 0.006, 30),
  "hyundai-porter-ii": buildMonthlySeries(...FORECAST_START, FORECAST_MONTHS, 1800, 0.006, 70),
};

/**
 * ponytail: derived from {@link MOCK_MONTHLY_SALES} so the two never drift apart —
 * "current" is the series' last generated month.
 */
export const MOCK_SALES_PREDICTION: Record<
  string,
  { currentMonth: number; prevMonth: number; halfYearAvg: number; prevYear: number }
> = Object.fromEntries(
  Object.entries(MOCK_MONTHLY_SALES).map(([key, series]) => {
    const last = series.length - 1;
    const lastSixValues = series.slice(Math.max(0, last - 5), last + 1).map((p) => p.sales);
    const halfYearAvg = Math.round(lastSixValues.reduce((sum, v) => sum + v, 0) / lastSixValues.length);
    const prevYearIndex = last - 12;
    return [
      key,
      {
        currentMonth: series[last].sales,
        prevMonth: series[Math.max(0, last - 1)].sales,
        halfYearAvg,
        prevYear: series[prevYearIndex >= 0 ? prevYearIndex : 0].sales,
      },
    ];
  }),
);
