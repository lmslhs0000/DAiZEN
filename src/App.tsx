import { useEffect, useState } from "react";
import { Sidebar, type TaskTab } from "@/components/Sidebar";
import { createEmptyTask, loadTasks, saveTasks, type DemandForecastTask } from "@/data/tasks";
import { TaskChart } from "@/pages/TaskChart";
import { TaskDetail } from "@/pages/TaskDetail";
import { WorkspaceList } from "@/pages/WorkspaceList";

type View = { name: "list" } | { name: "task"; taskId: string; tab: TaskTab };

function App() {
  const [view, setView] = useState<View>({ name: "list" });
  const [tasks, setTasks] = useState<DemandForecastTask[]>(() => loadTasks());

  useEffect(() => {
    saveTasks(tasks);
  }, [tasks]);

  function handleCreateTask() {
    const task = createEmptyTask();
    setTasks((prev) => [task, ...prev]);
    setView({ name: "task", taskId: task.id, tab: "input" });
  }

  function handleUpdateTask(id: string, patch: Partial<DemandForecastTask>) {
    setTasks((prev) => prev.map((t) => (t.id === id ? { ...t, ...patch, updatedAt: Date.now() } : t)));
  }

  function handleDeleteTask(id: string) {
    setTasks((prev) => prev.filter((t) => t.id !== id));
    if (view.name === "task" && view.taskId === id) {
      setView({ name: "list" });
    }
  }

  const activeTask = view.name === "task" ? tasks.find((t) => t.id === view.taskId) : undefined;

  return (
    <div className="min-h-dvh bg-paper-white">
      <header className="bg-midnight-ink px-6 py-4">
        <p className="text-[13px] font-medium text-sky-tint">베어링 수요 예측</p>
        <h1 className="text-[24px] font-semibold tracking-[-0.264px] text-pure-white">
          제조사·차종별 베어링 수요 예측
        </h1>
      </header>

      {view.name === "list" || !activeTask ? (
        <WorkspaceList
          tasks={tasks}
          onCreateTask={handleCreateTask}
          onOpenTask={(id) => setView({ name: "task", taskId: id, tab: "input" })}
          onDeleteTask={handleDeleteTask}
        />
      ) : (
        <div className="flex">
          <Sidebar
            active={view.tab}
            onNavigate={(tab) => setView({ name: "task", taskId: activeTask.id, tab })}
          />
          <div className="min-w-0 flex-1">
            {view.tab === "input" ? (
              <TaskDetail
                task={activeTask}
                onSave={(patch) => handleUpdateTask(activeTask.id, patch)}
                onBack={() => setView({ name: "list" })}
              />
            ) : (
              <TaskChart task={activeTask} onBack={() => setView({ name: "list" })} />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
