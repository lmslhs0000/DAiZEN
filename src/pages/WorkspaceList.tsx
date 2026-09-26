import { VehicleStatusBadge } from "@/components/VehicleStatusBadge";
import { formatTaskTimestamp, resolveVehicleStatus, taskTitle, type DemandForecastTask } from "@/data/tasks";

interface WorkspaceListProps {
  tasks: DemandForecastTask[];
  onCreateTask: () => void;
  onOpenTask: (id: string) => void;
  onDeleteTask: (id: string) => void;
}

export function WorkspaceList({ tasks, onCreateTask, onOpenTask, onDeleteTask }: WorkspaceListProps) {
  return (
    <main className="px-6 py-16">
      <div className="mx-auto max-w-[720px]">
        <div className="mb-6 flex items-start justify-between gap-4">
          <div>
            <h2 className="text-[20px] font-semibold text-onyx">수요예측 작업 목록</h2>
            <p className="mt-1 text-[14px] text-warm-gray">
              작업을 새로 만들거나, 저장해둔 작업을 이어서 진행하세요. 미완성 상태로도 저장할 수 있습니다.
            </p>
          </div>
          <button
            type="button"
            onClick={onCreateTask}
            className="h-11 shrink-0 rounded-lg bg-signal-blue px-5 text-[15px] font-medium text-pure-white transition-opacity hover:opacity-90"
          >
            + 새 작업 생성
          </button>
        </div>

        {tasks.length === 0 ? (
          <div className="rounded-xl border border-faint-line bg-pure-white p-8 text-center text-[14px] text-warm-gray">
            아직 저장된 작업이 없습니다. "새 작업 생성"으로 시작하세요.
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            {tasks.map((task) => {
              const status = resolveVehicleStatus(task);
              return (
                <div
                  key={task.id}
                  className="flex items-center justify-between rounded-xl border border-faint-line bg-pure-white p-4 shadow-subtle"
                >
                  <button type="button" onClick={() => onOpenTask(task.id)} className="flex-1 text-left">
                    <div className="mb-1 flex items-center gap-2">
                      <span className="text-[15px] font-semibold text-onyx">{taskTitle(task)}</span>
                      <VehicleStatusBadge status={status} />
                    </div>
                    <p className="text-[13px] text-warm-gray">최근 저장 · {formatTaskTimestamp(task.updatedAt)}</p>
                  </button>
                  <button
                    type="button"
                    onClick={() => onDeleteTask(task.id)}
                    className="ml-4 text-[13px] text-warm-gray hover:text-decrease"
                  >
                    삭제
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </main>
  );
}
