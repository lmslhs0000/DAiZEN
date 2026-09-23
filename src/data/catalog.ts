export interface VehicleModel {
  label: string;
  value: string;
  manufacturer: string;
}

export const VEHICLE_MODELS: VehicleModel[] = [
  { label: "Tesla Robotaxi", value: "tesla-robotaxi", manufacturer: "Tesla" },
];

/** ponytail: single-option lists for now — the vehicle-config table row selects these directly. */
export const COUNTRIES = [{ label: "한국", value: "kr" }];
export const MANUFACTURERS = [{ label: "Tesla", value: "tesla" }];
export const DRIVE_TYPES = [{ label: "EV", value: "ev" }];
export const MODELS = [{ label: "RoboTaxi", value: "robotaxi" }];

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
  "tesla-robotaxi": { currentMonth: 12400, prevMonth: 10800, halfYearAvg: 11200, prevYear: 9000 },
};

/** ponytail: same placeholder data, expressed as a monthly series for the trend chart. */
export const MOCK_MONTHLY_SALES: Record<string, { month: string; sales: number }[]> = {
  "tesla-robotaxi": [
    { month: "4월", sales: 10500 },
    { month: "5월", sales: 10900 },
    { month: "6월", sales: 11200 },
    { month: "7월", sales: 11400 },
    { month: "8월", sales: 10800 },
    { month: "9월", sales: 12400 },
  ],
};
