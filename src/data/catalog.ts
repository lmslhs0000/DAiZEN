export interface VehicleModel {
  label: string;
  value: string;
  manufacturer: string;
}

export const VEHICLE_MODELS: VehicleModel[] = [
  { label: "Tesla Cybercab", value: "tesla-cybercab", manufacturer: "Tesla" },
];

/**
 * ponytail: single-option lists for now — the vehicle-config table row selects these directly,
 * to be replaced by real API data. "ROBOTAXI" is a vehicle *role*, managed separately by the
 * backend — it is not a model name, so it must not be hardcoded here as one.
 */
export const COUNTRIES = [{ label: "한국", value: "kr" }];
export const MANUFACTURERS = [{ label: "Tesla", value: "tesla" }];
export const DRIVE_TYPES = [{ label: "EV", value: "ev" }];
export const MODELS = [{ label: "Cybercab", value: "cybercab" }];

export const BEARING_TYPES = [
  { label: "ENGINE", value: "engine" },
  { label: "TRANSMISSION", value: "transmission" },
  { label: "REDUCER", value: "reducer" },
  { label: "STEERING", value: "steering" },
];

/**
 * ponytail: hardcoded placeholder until the prediction backend is connected.
 * Monthly sales, or 6-month-average sales when monthly data is unavailable.
 */
export const MOCK_SALES_PREDICTION: Record<
  string,
  { currentMonth: number; prevMonth: number; halfYearAvg: number; prevYear: number }
> = {
  "tesla-cybercab": { currentMonth: 12400, prevMonth: 10800, halfYearAvg: 11200, prevYear: 9000 },
};

/**
 * ponytail: same placeholder data, expressed as a monthly series for the trend chart.
 * `month` is "YYYY-MM" so the series isn't tied to any fixed length — the chart renders
 * correctly whether the backend sends 1 month or up to 24 months of history.
 */
export const MOCK_MONTHLY_SALES: Record<string, { month: string; sales: number }[]> = {
  "tesla-cybercab": [
    { month: "2025-04", sales: 10500 },
    { month: "2025-05", sales: 10900 },
    { month: "2025-06", sales: 11200 },
    { month: "2025-07", sales: 11400 },
    { month: "2025-08", sales: 10800 },
    { month: "2025-09", sales: 12400 },
  ],
};
