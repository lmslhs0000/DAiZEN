import { COUNTRIES, DRIVE_TYPES, MANUFACTURERS, MODELS } from "@/data/catalog";

export interface DemandForecastTask {
  id: string;
  createdAt: number;
  updatedAt: number;
  country: string | null;
  manufacturer: string | null;
  driveType: string | null;
  model: string | null;
  quantities: Record<string, string>;
  naFlags: Record<string, boolean>;
}

const STORAGE_KEY = "demand-forecast-tasks";

export function loadTasks(): DemandForecastTask[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as DemandForecastTask[]) : [];
  } catch {
    return [];
  }
}

export function saveTasks(tasks: DemandForecastTask[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(tasks));
  } catch {
    // ponytail: localStorage can fail (private mode, quota) — workspace still works in-memory for this session.
  }
}

export function createEmptyTask(): DemandForecastTask {
  const now = Date.now();
  return {
    id: crypto.randomUUID(),
    createdAt: now,
    updatedAt: now,
    country: COUNTRIES[0]?.value ?? null,
    manufacturer: MANUFACTURERS[0]?.value ?? null,
    driveType: DRIVE_TYPES[0]?.value ?? null,
    model: MODELS[0]?.value ?? null,
    quantities: {},
    naFlags: {},
  };
}

/**
 * 백엔드가 관리하는 차량 상태 enum. RESOLVED가 아니면 어떤 정보가 빠졌는지 사용자에게
 * 구체적으로 알려준다 — raw enum 값을 그대로 노출하지 않는다.
 */
export type VehicleStatus =
  | "RESOLVED"
  | "AMBIGUOUS"
  | "APPLICABILITY_UNRESOLVED"
  | "IDENTITY_UNRESOLVED";

export const VEHICLE_STATUS_LABELS: Record<VehicleStatus, string> = {
  RESOLVED: "계산 가능",
  AMBIGUOUS: "구동방식 확인 필요",
  APPLICABILITY_UNRESOLVED: "국가/기간 적용정보 확인 필요",
  IDENTITY_UNRESOLVED: "차량 마스터 미등록",
};

/**
 * ponytail: 백엔드가 아직 실제 status 필드를 내려주지 않아, 현재 선택값으로부터
 * 프론트에서 근사 판정한다. 실제 API가 연결되면 이 함수 대신 서버가 준 값을 그대로 쓰면 된다.
 */
export function resolveVehicleStatus(selection: {
  country: string | null;
  manufacturer: string | null;
  driveType: string | null;
  model: string | null;
}): VehicleStatus {
  if (!selection.manufacturer || !selection.model) return "IDENTITY_UNRESOLVED";
  if (!selection.driveType) return "AMBIGUOUS";
  if (!selection.country) return "APPLICABILITY_UNRESOLVED";
  return "RESOLVED";
}

export function taskTitle(task: DemandForecastTask): string {
  const manufacturer = MANUFACTURERS.find((m) => m.value === task.manufacturer)?.label;
  const model = MODELS.find((m) => m.value === task.model)?.label;
  return manufacturer && model ? `${manufacturer} ${model}` : "제목 없는 작업";
}

export function formatTaskTimestamp(timestamp: number): string {
  return new Date(timestamp).toLocaleString("ko-KR", {
    month: "long",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}
