import { ClipboardList, LineChart } from "lucide-react";

export type TaskTab = "input" | "chart";

const MENU_ITEMS: { id: TaskTab; label: string; icon: typeof ClipboardList }[] = [
  { id: "input", label: "베어링 입력", icon: ClipboardList },
  { id: "chart", label: "판매량 그래프", icon: LineChart },
];

interface SidebarProps {
  active: TaskTab;
  onNavigate: (tab: TaskTab) => void;
}

export function Sidebar({ active, onNavigate }: SidebarProps) {
  return (
    <nav className="w-[200px] shrink-0 border-r border-faint-line px-3 py-8">
      <ul className="flex flex-col gap-1">
        {MENU_ITEMS.map((item) => {
          const isActive = item.id === active;
          return (
            <li key={item.id}>
              <button
                type="button"
                onClick={() => onNavigate(item.id)}
                aria-current={isActive ? "page" : undefined}
                className={`flex w-full items-center gap-2 rounded-lg px-3 py-2 text-[14px] font-medium transition-colors ${
                  isActive
                    ? "bg-paper-white text-signal-blue"
                    : "text-warm-gray hover:bg-paper-white hover:text-onyx"
                }`}
              >
                <item.icon className="h-4 w-4" aria-hidden="true" />
                {item.label}
              </button>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
