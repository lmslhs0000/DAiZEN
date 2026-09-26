import { VEHICLE_STATUS_LABELS, type VehicleStatus } from "@/data/tasks";

export function VehicleStatusBadge({ status }: { status: VehicleStatus }) {
  const isResolved = status === "RESOLVED";
  return (
    <span
      className={`rounded-full bg-paper-white px-2 py-0.5 text-[11px] font-medium ${
        isResolved ? "text-signal-blue" : "text-decrease"
      }`}
    >
      {VEHICLE_STATUS_LABELS[status]}
    </span>
  );
}
