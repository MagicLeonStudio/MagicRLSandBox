import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, HelpCircle } from "lucide-react";
import { games, getAlgorithmsForGame } from "@/data/algorithms";
import type { Algorithm } from "@/data/algorithms";

interface NavbarProps {
  selectedGame: string;
  selectedAlgorithm: string;
  algorithms: Algorithm[];
  onGameChange: (gameId: string) => void;
  onAlgorithmChange: (algorithmId: string) => void;
  onShowHelp: () => void;
}

function Dropdown({
  label,
  value,
  options,
  onChange,
  renderOption,
}: {
  label: string;
  value: string;
  options: { id: string; name: string }[];
  onChange: (id: string) => void;
  renderOption?: (option: { id: string; name: string }) => React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const selected = options.find((o) => o.id === value);

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 bg-bg-input border border-[rgba(139,92,246,0.2)] rounded-full px-4 py-2 text-sm transition-all duration-200 hover:border-accent-purple/50"
      >
        <span className="text-text-muted text-xs">{label}</span>
        <span className="text-text-primary text-[13px] font-medium">
          {selected?.name ?? value}
        </span>
        <motion.span
          animate={{ rotate: open ? 180 : 0 }}
          transition={{ duration: 0.2 }}
        >
          <ChevronDown size={14} className="text-text-muted" />
        </motion.span>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className="absolute top-full mt-2 right-0 bg-bg-elevated border border-[rgba(139,92,246,0.2)] rounded-lg p-1 min-w-[200px] z-[200] backdrop-blur-xl shadow-xl"
          >
            {options.map((option) => (
              <button
                key={option.id}
                onClick={() => {
                  onChange(option.id);
                  setOpen(false);
                }}
                className={`w-full text-left px-4 py-2.5 rounded-md text-sm transition-all duration-150 flex items-center gap-2 ${
                  option.id === value
                    ? "text-accent-purple border-l-2 border-accent-purple bg-[rgba(139,92,246,0.08)]"
                    : "text-text-secondary hover:bg-[rgba(139,92,246,0.08)]"
                }`}
              >
                {renderOption ? renderOption(option) : option.name}
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function Navbar({
  selectedGame,
  selectedAlgorithm,
  algorithms,
  onGameChange,
  onAlgorithmChange,
  onShowHelp,
}: NavbarProps) {
  const gameAlgorithms = getAlgorithmsForGame(selectedGame);

  return (
    <header
      className="sticky top-0 left-0 right-0 z-[100] h-14 flex items-center justify-between px-6"
      style={{
        backgroundColor: "rgba(18, 18, 26, 0.85)",
        backdropFilter: "blur(16px)",
        borderBottom: "1px solid rgba(139, 92, 246, 0.15)",
      }}
    >
      {/* Left: Brand */}
      <div className="flex items-center gap-2">
        <CircuitNodeIcon />
        <span
          className="text-text-primary font-bold text-base tracking-[0.08em] font-inter"
          style={{ letterSpacing: "0.08em" }}
        >
          RL ARENA
        </span>
      </div>

      {/* Right: Controls */}
      <div className="flex items-center gap-2">
        <Dropdown
          label="Game:"
          value={selectedGame}
          options={games.map((g) => ({ id: g.id, name: g.name }))}
          onChange={onGameChange}
        />

        <Dropdown
          label="Algorithm:"
          value={selectedAlgorithm}
          options={gameAlgorithms.map((a) => ({
            id: a.id,
            name: a.name,
          }))}
          onChange={onAlgorithmChange}
          renderOption={(option) => {
            const algo = algorithms.find((a) => a.id === option.id);
            const badgeColors: Record<string, string> = {
              Classic: "text-accent-yellow bg-[rgba(250,204,21,0.15)] border-[rgba(250,204,21,0.3)]",
              "Deep RL": "text-accent-purple bg-[rgba(139,92,246,0.15)] border-[rgba(139,92,246,0.3)]",
              Modern: "text-accent-watermelon bg-[rgba(255,107,107,0.15)] border-[rgba(255,107,107,0.3)]",
            };
            return (
              <div className="flex items-center gap-2">
                <span className="font-medium">{option.name}</span>
                {algo && (
                  <span
                    className={`text-[10px] font-semibold px-1.5 py-0.5 rounded-full border ${
                      badgeColors[algo.category] ?? ""
                    }`}
                  >
                    {algo.category}
                  </span>
                )}
              </div>
            );
          }}
        />

        <button
          onClick={onShowHelp}
          className="p-2 text-text-muted hover:text-accent-purple hover:scale-110 transition-all duration-150"
        >
          <HelpCircle size={20} />
        </button>
      </div>
    </header>
  );
}

function CircuitNodeIcon() {
  return (
    <svg
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className="text-accent-purple"
    >
      <circle cx="12" cy="12" r="2" fill="currentColor" />
      <circle cx="5" cy="5" r="1.5" fill="currentColor" opacity="0.6" />
      <circle cx="19" cy="5" r="1.5" fill="currentColor" opacity="0.6" />
      <circle cx="5" cy="19" r="1.5" fill="currentColor" opacity="0.6" />
      <circle cx="19" cy="19" r="1.5" fill="currentColor" opacity="0.6" />
      <line
        x1="12"
        y1="10"
        x2="12"
        y2="6"
        stroke="currentColor"
        strokeWidth="1.5"
        opacity="0.4"
      />
      <line
        x1="12"
        y1="14"
        x2="12"
        y2="18"
        stroke="currentColor"
        strokeWidth="1.5"
        opacity="0.4"
      />
      <line
        x1="10"
        y1="12"
        x2="6"
        y2="12"
        stroke="currentColor"
        strokeWidth="1.5"
        opacity="0.4"
      />
      <line
        x1="14"
        y1="12"
        x2="18"
        y2="12"
        stroke="currentColor"
        strokeWidth="1.5"
        opacity="0.4"
      />
      <line
        x1="6.5"
        y1="6.5"
        x2="10"
        y2="10"
        stroke="currentColor"
        strokeWidth="1"
        opacity="0.3"
      />
      <line
        x1="17.5"
        y1="6.5"
        x2="14"
        y2="10"
        stroke="currentColor"
        strokeWidth="1"
        opacity="0.3"
      />
      <line
        x1="6.5"
        y1="17.5"
        x2="10"
        y2="14"
        stroke="currentColor"
        strokeWidth="1"
        opacity="0.3"
      />
      <line
        x1="17.5"
        y1="17.5"
        x2="14"
        y2="14"
        stroke="currentColor"
        strokeWidth="1"
        opacity="0.3"
      />
    </svg>
  );
}
