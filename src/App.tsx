import { useState } from "react";
import { Sidebar, type View } from "@/components/Sidebar";
import { BearingPlanner } from "@/pages/BearingPlanner";
import { ChartPage } from "@/pages/ChartPage";

function App() {
  const [view, setView] = useState<View>("planner");

  return (
    <div className="min-h-dvh bg-paper-white">
      <header className="bg-midnight-ink px-6 py-4">
        <p className="text-[13px] font-medium text-sky-tint">베어링 수요 예측</p>
        <h1 className="text-[24px] font-semibold tracking-[-0.264px] text-pure-white">
          제조사·차종별 베어링 수요 예측
        </h1>
      </header>

      <div className="flex">
        <Sidebar active={view} onNavigate={setView} />
        <div className="min-w-0 flex-1">
          {view === "planner" ? <BearingPlanner /> : <ChartPage />}
        </div>
      </div>
    </div>
  );
}

export default App;
