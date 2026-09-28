import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, RotateCcw, X, ChevronRight, BarChart3 } from "lucide-react";
import ArenaNavbar from "@/components/ArenaNavbar";
import RewardChart from "@/components/RewardChart";
import LossChart from "@/components/LossChart";
import { algorithms, getAlgorithmsForGame } from "@/data/algorithms";
import type { Algorithm } from "@/data/algorithms";
import { GridWorldEngine } from "@/engines/GridWorldEngine";
import { CartPoleEngine } from "@/engines/CartPoleEngine";
import { SnakeEngine } from "@/engines/SnakeEngine";
import { QLearning } from "@/algorithms/QLearning";
import { SARSA } from "@/algorithms/SARSA";
import { DQN } from "@/algorithms/DQN";
import { A3C } from "@/algorithms/A3C";
import { REINFORCE } from "@/algorithms/REINFORCE";
import { ActorCritic } from "@/algorithms/ActorCritic";
import { PPO } from "@/algorithms/PPO";
import { TRPO } from "@/algorithms/TRPO";

type GameId = "grid-world" | "cart-pole" | "snake";

// Algorithm factory
function createAlgorithm(algorithm: Algorithm, stateSize: number, actionSize: number) {
  const p = algorithm.parameters;
  const get = (name: string) => p.find((x) => x.name === name)?.default ?? 0;

  switch (algorithm.id) {
    case "q-learning":
      return new QLearning(stateSize, actionSize, {
        learningRate: get("learningRate"),
        epsilon: get("epsilon"),
        discount: get("discount"),
        epsilonDecay: get("epsilonDecay"),
      });
    case "sarsa":
      return new SARSA(stateSize, actionSize, {
        learningRate: get("learningRate"),
        epsilon: get("epsilon"),
        discount: get("discount"),
        epsilonDecay: get("epsilonDecay"),
      });
    case "dqn-grid":
    case "dqn-snake":
      return new DQN(stateSize, actionSize, {
        learningRate: get("learningRate"),
        epsilon: get("epsilon"),
        batchSize: get("batchSize"),
        replayBuffer: get("replayBuffer"),
        targetUpdate: get("targetUpdate"),
      });
    case "a3c-grid":
    case "a3c-snake":
      return new A3C(stateSize, actionSize, {
        learningRate: get("learningRate"),
        gamma: get("gamma"),
        entropyCoef: get("entropyCoef"),
        valueCoef: get("valueCoef"),
      });
    case "reinforce":
      return new REINFORCE(stateSize, actionSize, {
        learningRate: get("learningRate"),
        gamma: get("gamma"),
      });
    case "actor-critic":
      return new ActorCritic(stateSize, actionSize, {
        learningRate: get("learningRate"),
        gamma: get("gamma"),
        entropyCoef: get("entropyCoef"),
      });
    case "ppo":
      return new PPO(stateSize, actionSize, {
        learningRate: get("learningRate"),
        gamma: get("gamma"),
        clipEpsilon: get("clipEpsilon"),
        entropyCoef: get("entropyCoef"),
      });
    case "trpo":
      return new TRPO(stateSize, actionSize, {
        learningRate: get("learningRate"),
        gamma: get("gamma"),
        delta: get("maxKL"),
      });
    default:
      return new QLearning(stateSize, actionSize, {
        learningRate: 0.1,
        epsilon: 0.3,
        discount: 0.95,
      });
  }
}

export default function Playground() {
  const [selectedGame, setSelectedGame] = useState<GameId>("grid-world");
  const [selectedAlgorithm, setSelectedAlgorithm] = useState("q-learning");
  const [isTraining, setIsTraining] = useState(false);
  const [currentEpisode, setCurrentEpisode] = useState(0);
  const [totalReward, setTotalReward] = useState(0);
  const [currentLoss, setCurrentLoss] = useState(0);
  const [rewardHistory, setRewardHistory] = useState<number[]>([]);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [stepCount, setStepCount] = useState(0);
  const [epsilon, setEpsilon] = useState(0.3);
  const [showHelp, setShowHelp] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [modalAlgorithm, setModalAlgorithm] = useState<Algorithm | null>(null);
  const [paramValues, setParamValues] = useState<Record<string, number>>({});

  // Evaluation state
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evalEpisode, setEvalEpisode] = useState(0);
  const [evalTotalEpisodes, setEvalTotalEpisodes] = useState(10);
  const [evalResults, setEvalResults] = useState<{ episode: number; reward: number; steps: number }[]>([]);
  const [evalAverageReward, setEvalAverageReward] = useState(0);
  const [evalSuccessRate, setEvalSuccessRate] = useState(0);
  const [evalMaxReward, setEvalMaxReward] = useState(0);
  const [evalMinReward, setEvalMinReward] = useState(0);
  const [evalAverageSteps, setEvalAverageSteps] = useState(0);
  const [selectedEvalEpisodes, setSelectedEvalEpisodes] = useState(10);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number | null>(null);
  const engineRef = useRef<GridWorldEngine | CartPoleEngine | SnakeEngine | null>(null);
  const algoRef = useRef<QLearning | SARSA | DQN | A3C | REINFORCE | ActorCritic | PPO | TRPO | null>(null);
  const isTrainingRef = useRef(false);
  const episodeStepsRef = useRef(0);
  const episodeRewardRef = useRef(0);

  const gameAlgorithms = getAlgorithmsForGame(selectedGame);
  const currentAlgorithm = algorithms.find((a) => a.id === selectedAlgorithm) ?? algorithms[0];

  // Initialize param values when algorithm changes
  useEffect(() => {
    const defaults: Record<string, number> = {};
    currentAlgorithm.parameters.forEach((p) => {
      defaults[p.name] = p.default;
    });
    setParamValues(defaults);
  }, [selectedAlgorithm]);

  // Initialize game engine
  const initEngine = useCallback(() => {
    let engine;
    switch (selectedGame) {
      case "grid-world":
        engine = new GridWorldEngine();
        break;
      case "cart-pole":
        engine = new CartPoleEngine();
        break;
      case "snake":
        engine = new SnakeEngine();
        break;
      default:
        engine = new GridWorldEngine();
    }
    engineRef.current = engine;
    return engine;
  }, [selectedGame]);

  // Initialize algorithm
  const initAlgorithm = useCallback(
    (engine: GridWorldEngine | CartPoleEngine | SnakeEngine) => {
      const algo = createAlgorithm(
        currentAlgorithm,
        engine.getStateSize(),
        engine.getActionSize()
      );
      algoRef.current = algo;
      return algo;
    },
    [currentAlgorithm]
  );

  // Get canvas dimensions based on game
  const getCanvasDimensions = useCallback(() => {
    switch (selectedGame) {
      case "grid-world":
        return { width: 500, height: 500 };
      case "cart-pole":
        return { width: 700, height: 350 };
      case "snake":
        return { width: 500, height: 500 };
      default:
        return { width: 500, height: 500 };
    }
  }, [selectedGame]);

  // Render game on canvas
  const renderGame = useCallback(() => {
    const canvas = canvasRef.current;
    const engine = engineRef.current;
    if (!canvas || !engine) return;

    const dims = getCanvasDimensions();
    canvas.width = dims.width;
    canvas.height = dims.height;

    engine.render(canvas);
  }, [getCanvasDimensions]);

  // Initialize everything
  const resetAll = useCallback(() => {
    const engine = initEngine();
    initAlgorithm(engine);
    engine.reset();
    setCurrentEpisode(0);
    setTotalReward(0);
    setCurrentLoss(0);
    setRewardHistory([]);
    setLossHistory([]);
    setStepCount(0);
    setEpsilon(0.3);
    episodeStepsRef.current = 0;
    episodeRewardRef.current = 0;

    requestAnimationFrame(renderGame);
  }, [initEngine, initAlgorithm, renderGame]);

  // Initial setup
  useEffect(() => {
    resetAll();
  }, [resetAll]);

  // Re-render when game changes
  useEffect(() => {
    // Auto-select first algorithm for new game
    const firstAlgo = gameAlgorithms[0];
    if (firstAlgo && !gameAlgorithms.find((a) => a.id === selectedAlgorithm)) {
      setSelectedAlgorithm(firstAlgo.id);
    }
  }, [selectedGame, gameAlgorithms, selectedAlgorithm]);

  // Training loop
  useEffect(() => {
    if (!isTraining) {
      if (animFrameRef.current !== null) {
        cancelAnimationFrame(animFrameRef.current);
        animFrameRef.current = null;
      }
      return;
    }

    let stepsPerFrame = 0;
    const MAX_STEPS_PER_FRAME = selectedGame === "cart-pole" ? 16 : 8;

    const loop = () => {
      if (!isTrainingRef.current || !engineRef.current || !algoRef.current) {
        animFrameRef.current = null;
        return;
      }

      const engine = engineRef.current;
      const algo = algoRef.current;
      stepsPerFrame = 0;

      while (stepsPerFrame < MAX_STEPS_PER_FRAME) {
        const state = engine.getObservation();
        const action = algo.selectAction(state, epsilon);
        const result = engine.step(action as 0 & 1 & 2 & 3);

        const loss = algo.train(state, action, result.reward, result.state, result.done);
        episodeStepsRef.current++;
        episodeRewardRef.current += result.reward;
        stepsPerFrame++;

        if (result.done) {
          // Episode complete
          setCurrentEpisode((prev) => prev + 1);
          setTotalReward(Math.round(episodeRewardRef.current));
          setCurrentLoss(loss);
          setStepCount((prev) => prev + episodeStepsRef.current);

          // @ts-expect-error accessing epsilon property dynamically
          const currentEpsilon = algo.epsilon ?? epsilon;
          setEpsilon(typeof currentEpsilon === "number" ? currentEpsilon : 0.01);

          setRewardHistory((prev) => {
            const next = [...prev, Math.round(episodeRewardRef.current)];
            if (next.length > 200) return next.slice(-200);
            return next;
          });
          setLossHistory((prev) => {
            const next = [...prev, loss];
            if (next.length > 200) return next.slice(-200);
            return next;
          });

          engine.reset();
          episodeStepsRef.current = 0;
          episodeRewardRef.current = 0;
          break;
        }
      }

      renderGame();
      animFrameRef.current = requestAnimationFrame(loop);
    };

    isTrainingRef.current = true;
    animFrameRef.current = requestAnimationFrame(loop);

    return () => {
      if (animFrameRef.current !== null) {
        cancelAnimationFrame(animFrameRef.current);
        animFrameRef.current = null;
      }
    };
  }, [isTraining, epsilon, selectedGame, renderGame]);

  // Handle game change
  const handleGameChange = useCallback(
    (gameId: string) => {
      setIsTraining(false);
      isTrainingRef.current = false;
      setSelectedGame(gameId as GameId);

      // Will trigger resetAll via useEffect
      setTimeout(() => {
        const engine = initEngine();
        const algos = getAlgorithmsForGame(gameId);
        const algo = algos[0];
        if (algo) {
          setSelectedAlgorithm(algo.id);
          const algoInstance = createAlgorithm(algo, engine.getStateSize(), engine.getActionSize());
          algoRef.current = algoInstance;
        }
        engineRef.current = engine;
        engine.reset();
        setCurrentEpisode(0);
        setTotalReward(0);
        setCurrentLoss(0);
        setRewardHistory([]);
        setLossHistory([]);
        setStepCount(0);
        setEpsilon(0.3);
        episodeStepsRef.current = 0;
        episodeRewardRef.current = 0;
        // Clear eval results
        setEvalResults([]);
        setEvalAverageReward(0);
        setEvalSuccessRate(0);
        setEvalMaxReward(0);
        setEvalMinReward(0);
        setEvalAverageSteps(0);
        requestAnimationFrame(renderGame);
      }, 0);
    },
    [initEngine, renderGame]
  );

  // Handle algorithm change
  const handleAlgorithmChange = useCallback(
    (algoId: string) => {
      setIsTraining(false);
      isTrainingRef.current = false;
      setSelectedAlgorithm(algoId);

      const engine = engineRef.current;
      if (!engine) return;

      const algo = algorithms.find((a) => a.id === algoId);
      if (algo) {
        const algoInstance = createAlgorithm(algo, engine.getStateSize(), engine.getActionSize());
        algoRef.current = algoInstance;
      }

      engine.reset();
      setCurrentEpisode(0);
      setTotalReward(0);
      setCurrentLoss(0);
      setRewardHistory([]);
      setLossHistory([]);
      setStepCount(0);
      setEpsilon(0.3);
      episodeStepsRef.current = 0;
      episodeRewardRef.current = 0;
      // Clear eval results
      setEvalResults([]);
      setEvalAverageReward(0);
      setEvalSuccessRate(0);
      setEvalMaxReward(0);
      setEvalMinReward(0);
      setEvalAverageSteps(0);
      renderGame();
    },
    [renderGame]
  );

  const handleStartStop = useCallback(() => {
    setIsTraining((prev) => {
      const next = !prev;
      isTrainingRef.current = next;
      if (next) {
        // Clear eval results when starting training
        setEvalResults([]);
        setEvalAverageReward(0);
        setEvalSuccessRate(0);
        setEvalMaxReward(0);
        setEvalMinReward(0);
        setEvalAverageSteps(0);
      }
      return next;
    });
  }, []);

  const handleReset = useCallback(() => {
    setIsTraining(false);
    isTrainingRef.current = false;
    resetAll();
    // Clear eval results
    setEvalResults([]);
    setEvalAverageReward(0);
    setEvalSuccessRate(0);
    setEvalMaxReward(0);
    setEvalMinReward(0);
    setEvalAverageSteps(0);
  }, [resetAll]);

  const handleEvaluate = useCallback(async () => {
    if (!engineRef.current || !algoRef.current) return;
    if (isTraining) return;

    const totalEpisodes = selectedEvalEpisodes;
    const engine = engineRef.current;
    const algo = algoRef.current;

    setIsEvaluating(true);
    setEvalTotalEpisodes(totalEpisodes);
    setEvalEpisode(0);
    setEvalResults([]);
    setEvalAverageReward(0);
    setEvalSuccessRate(0);
    setEvalMaxReward(0);
    setEvalMinReward(0);
    setEvalAverageSteps(0);

    const results: { episode: number; reward: number; steps: number }[] = [];
    let totalRewardSum = 0;
    let successes = 0;
    let totalSteps = 0;
    let maxReward = -Infinity;
    let minReward = Infinity;
    const MAX_EVAL_STEPS_PER_EPISODE = 500; // absolute safety limit

    for (let ep = 0; ep < totalEpisodes; ep++) {
      setEvalEpisode(ep + 1);
      engine.reset();
      let episodeReward = 0;
      let episodeSteps = 0;
      let done = false;

      while (!done && episodeSteps < MAX_EVAL_STEPS_PER_EPISODE) {
        const state = engine.getObservation();
        const action = algo.selectAction(state, 0); // greedy: epsilon = 0
        const result = engine.step(action as 0 & 1 & 2 & 3);
        episodeReward += result.reward;
        episodeSteps++;
        done = result.done;

        renderGame();
        await new Promise((resolve) => setTimeout(resolve, 200)); // 200ms per step
      }

      // If exited due to step limit, penalize and mark done
      if (!done && episodeSteps >= MAX_EVAL_STEPS_PER_EPISODE) {
        done = true;
        episodeReward -= 10; // timeout penalty
      }

      // Determine success based on game
      let isSuccess = false;
      if (selectedGame === "grid-world") {
        // Success = reached target (positive total reward means target reached)
        isSuccess = episodeReward > 0;
      } else if (selectedGame === "cart-pole") {
        // Success = balanced for more than 100 steps
        isSuccess = episodeSteps > 100;
      } else if (selectedGame === "snake") {
        // Success = ate at least 1 food (score tracked in engine state)
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        isSuccess = ((engine as any).state?.score ?? 0) >= 1;
      }

      if (isSuccess) successes++;
      totalRewardSum += episodeReward;
      totalSteps += episodeSteps;
      maxReward = Math.max(maxReward, episodeReward);
      minReward = Math.min(minReward, episodeReward);

      const result = { episode: ep + 1, reward: episodeReward, steps: episodeSteps };
      results.push(result);

      setEvalResults([...results]);
      setEvalAverageReward(totalRewardSum / (ep + 1));
      setEvalSuccessRate((successes / (ep + 1)) * 100);
      setEvalMaxReward(maxReward);
      setEvalMinReward(minReward);
      setEvalAverageSteps(totalSteps / (ep + 1));
    }

    setIsEvaluating(false);
  }, [isTraining, selectedEvalEpisodes, selectedGame, renderGame]);

  const handleShowModal = useCallback(() => {
    setModalAlgorithm(currentAlgorithm);
    setShowModal(true);
  }, [currentAlgorithm]);

  const handleParamChange = useCallback(
    (name: string, value: number) => {
      setParamValues((prev) => ({ ...prev, [name]: value }));

      // Update algorithm instance
      if (algoRef.current) {
        // @ts-expect-error dynamic property access
        algoRef.current[name] = value;
      }
    },
    []
  );

  const categoryBadgeColors: Record<string, string> = {
    Classic: "text-accent-yellow bg-[rgba(250,204,21,0.15)] border-[rgba(250,204,21,0.3)]",
    "Deep RL": "text-accent-purple bg-[rgba(139,92,246,0.15)] border-[rgba(139,92,246,0.3)]",
    Modern: "text-accent-watermelon bg-[rgba(255,107,107,0.15)] border-[rgba(255,107,107,0.3)]",
  };

  const gameDisplayNames: Record<string, string> = {
    "grid-world": "GRID WORLD",
    "cart-pole": "CART POLE",
    snake: "SNAKE",
  };

  return (
    <div className="min-h-[100dvh] bg-bg-primary">
      <ArenaNavbar
        selectedGame={selectedGame}
        selectedAlgorithm={selectedAlgorithm}
        algorithms={algorithms}
        onGameChange={handleGameChange}
        onAlgorithmChange={handleAlgorithmChange}
        onShowHelp={() => setShowHelp(true)}
      />

      <div className="flex flex-col lg:flex-row h-[calc(100dvh-56px)]">
        {/* Game Canvas Section */}
        <div className="flex-1 flex items-center justify-center p-4 relative">
          <div className="flex flex-col items-center gap-3">
            {/* Ingame HUD - moved above canvas */}
            <div
              className="flex items-center gap-4 px-4 py-1.5 rounded-full"
              style={{
                backgroundColor: "rgba(10, 10, 15, 0.85)",
                backdropFilter: "blur(8px)",
                border: "1px solid rgba(255, 255, 255, 0.05)",
              }}
            >
              <span className="font-mono text-[11px] text-text-secondary tracking-[0.1em]">
                {gameDisplayNames[selectedGame]}
              </span>
              <span
                className="w-px h-3"
                style={{ backgroundColor: "rgba(255,255,255,0.1)" }}
              />
              <span className="font-mono text-[11px] text-accent-yellow">
                EP: {currentEpisode}
              </span>
              <span className="font-mono text-[11px] text-text-primary">
                R: {totalReward}
              </span>
              {selectedGame === "grid-world" && (
                <span className="font-mono text-[11px] text-accent-purple">
                  &epsilon;: {epsilon.toFixed(2)}
                </span>
              )}
            </div>

            <motion.div
              key={selectedGame}
              initial={{ opacity: 0, scale: 0.97 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.97 }}
              transition={{ duration: 0.3 }}
              className="relative bg-bg-surface rounded-xl border border-[rgba(139,92,246,0.08)] shadow-glow-game p-4"
            >
              {/* Algorithm Badge */}
              <AnimatePresence mode="wait">
                <motion.div
                  key={selectedAlgorithm}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.2 }}
                  className="absolute top-3 left-3 z-10"
                >
                  <div
                    className="bg-[rgba(139,92,246,0.15)] backdrop-blur-sm border border-[rgba(139,92,246,0.3)] rounded px-3 py-1"
                  >
                    <span className="font-mono text-[10px] font-semibold text-accent-purple">
                      {currentAlgorithm.name}
                    </span>
                  </div>
                </motion.div>
              </AnimatePresence>

              <canvas
              ref={canvasRef}
              style={{
                imageRendering: "pixelated",
                display: "block",
                maxWidth: selectedGame === "cart-pole" ? 700 : 500,
                maxHeight: selectedGame === "cart-pole" ? 350 : 500,
                width: "100%",
                height: "auto",
              }}
            />
          </motion.div>
        </div>
        </div>

        {/* Control Panel */}
        <motion.aside
          initial={{ opacity: 0, x: 40 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4, delay: 0.1 }}
          className="w-full lg:w-[520px] lg:min-w-[520px] bg-bg-surface border-l border-[rgba(139,92,246,0.1)] p-5 overflow-y-auto flex flex-col gap-5"
          style={{ height: "calc(100dvh - 56px)" }}
        >
          {/* Algorithm Info Card */}
          <AnimatePresence mode="wait">
            <motion.div
              key={selectedAlgorithm}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="bg-bg-elevated rounded-lg p-4 border border-[rgba(139,92,246,0.15)]"
            >
              {/* Category Badge */}
              <span
                className={`inline-block text-[11px] font-semibold px-2.5 py-0.5 rounded-full border ${
                  categoryBadgeColors[currentAlgorithm.category] ?? ""
                }`}
              >
                {currentAlgorithm.category}
              </span>

              {/* Algorithm Name */}
              <h2 className="text-text-primary text-lg font-bold mt-2">
                {currentAlgorithm.name}
              </h2>

              {/* Full Name */}
              <p className="text-text-secondary text-[13px] italic mt-0.5">
                {currentAlgorithm.fullName}
              </p>

              {/* Year Badge */}
              <span className="inline-block mt-1 text-[11px] text-text-muted bg-bg-input px-2 py-0.5 rounded">
                {currentAlgorithm.year}
              </span>

              {/* Description */}
              <p className="text-text-secondary text-[13px] leading-relaxed mt-3">
                {currentAlgorithm.description}
              </p>

              {/* Read More Link */}
              <button
                onClick={handleShowModal}
                className="mt-3 text-accent-purple text-[13px] font-medium flex items-center gap-1 group transition-colors duration-150 hover:text-accent-yellow"
              >
                Read More
                <ChevronRight
                  size={14}
                  className="transition-transform duration-150 group-hover:translate-x-1"
                />
              </button>
            </motion.div>
          </AnimatePresence>

          {/* Parameter Controls */}
          <AnimatePresence mode="wait">
            <motion.div
              key={`params-${selectedAlgorithm}`}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2, delay: 0.1 }}
            >
              <h3 className="text-text-primary text-sm font-semibold tracking-[0.02em] mb-4">
                Parameters
              </h3>
              <div className="flex flex-col gap-4">
                {currentAlgorithm.parameters.map((param) => (
                  <div key={param.name}>
                    <div className="flex justify-between mb-1.5">
                      <span className="text-text-secondary text-xs capitalize">
                        {param.name.replace(/([A-Z])/g, " $1").trim()}
                      </span>
                      <span className="font-mono text-xs text-accent-purple font-semibold">
                        {paramValues[param.name]?.toFixed(
                          param.step >= 1 ? 0 : param.step >= 0.1 ? 2 : 4
                        ) ?? param.default}
                      </span>
                    </div>
                    <input
                      type="range"
                      min={param.min}
                      max={param.max}
                      step={param.step}
                      value={paramValues[param.name] ?? param.default}
                      onChange={(e) =>
                        handleParamChange(param.name, parseFloat(e.target.value))
                      }
                      className="w-full h-1 bg-bg-input rounded-full appearance-none cursor-pointer accent-accent-purple
                        [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-accent-purple [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-[#A78BFA] [&::-webkit-slider-thumb]:shadow-glow-purple [&::-webkit-slider-thumb]:transition-transform [&::-webkit-slider-thumb]:hover:scale-125 [&::-webkit-slider-thumb]:active:scale-95
                        [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-accent-purple [&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-[#A78BFA] [&::-moz-range-thumb]:shadow-glow-purple"
                      style={{
                        background: `linear-gradient(to right, #8B5CF6 0%, #A78BFA ${
                          (((paramValues[param.name] ?? param.default) - param.min) /
                            (param.max - param.min)) *
                          100
                        }%, #1E1E2E ${
                          (((paramValues[param.name] ?? param.default) - param.min) /
                            (param.max - param.min)) *
                          100
                        }%, #1E1E2E 100%)`,
                      }}
                    />
                  </div>
                ))}
              </div>
            </motion.div>
          </AnimatePresence>

          {/* Action Buttons */}
          <div className="flex gap-2">
            <button
              onClick={handleStartStop}
              className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg font-semibold text-sm text-white transition-all duration-200 ${
                isTraining
                  ? "bg-gradient-to-br from-accent-purple to-[#7C3AED] animate-pulse"
                  : "bg-gradient-to-br from-accent-purple to-[#7C3AED] hover:brightness-115"
              }`}
              style={
                isTraining
                  ? {
                      boxShadow:
                        "0 0 20px rgba(139, 92, 246, 0.3), 0 0 30px rgba(139, 92, 246, 0.5)",
                      animation: "pulse 1.5s infinite",
                    }
                  : {}
              }
            >
              {isTraining ? (
                <>
                  <Pause size={16} /> Stop
                </>
              ) : (
                <>
                  <Play size={16} /> Start Training
                </>
              )}
            </button>
            <button
              onClick={handleReset}
              className="px-4 py-2.5 rounded-lg border border-[rgba(156,163,175,0.3)] text-text-secondary text-sm hover:border-accent-purple hover:text-accent-purple transition-all duration-200 flex items-center gap-2"
            >
              <RotateCcw size={16} /> Reset
            </button>
          </div>

          {/* Evaluation Controls */}
          <div className="bg-bg-elevated rounded-lg p-4 border border-[rgba(139,92,246,0.08)]">
            <h3 className="text-text-primary text-[13px] font-semibold mb-3">
              Evaluation
            </h3>
            {/* Quick select buttons */}
            <div className="flex gap-1.5 mb-2">
              {[
                { label: "10", value: 10 },
                {
                  label: `${Math.max(10, Math.round(currentEpisode * 0.1))}`,
                  value: Math.max(10, Math.round(currentEpisode * 0.1)),
                },
                {
                  label: `${Math.max(10, Math.round(currentEpisode * 0.5))}`,
                  value: Math.max(10, Math.round(currentEpisode * 0.5)),
                },
                {
                  label: `${Math.max(10, currentEpisode)}`,
                  value: Math.max(10, currentEpisode),
                },
              ].map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => setSelectedEvalEpisodes(opt.value)}
                  disabled={isTraining || isEvaluating}
                  className={`flex-1 text-[10px] font-mono py-1 rounded border transition-all ${
                    selectedEvalEpisodes === opt.value
                      ? "bg-[rgba(139,92,246,0.2)] border-accent-purple text-accent-purple"
                      : "bg-bg-input border-[rgba(139,92,246,0.1)] text-text-secondary hover:border-[rgba(139,92,246,0.3)]"
                  } disabled:opacity-50`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-2 mb-3">
              <input
                type="number"
                min={1}
                max={Math.max(10, currentEpisode)}
                value={selectedEvalEpisodes}
                onChange={(e) => {
                  const v = parseInt(e.target.value);
                  if (!isNaN(v) && v >= 1)
                    setSelectedEvalEpisodes(
                      Math.min(v, Math.max(10, currentEpisode))
                    );
                }}
                disabled={isTraining || isEvaluating}
                className="w-20 bg-bg-input text-text-primary text-sm rounded-lg px-3 py-2 border border-[rgba(139,92,246,0.15)] focus:outline-none focus:border-accent-purple disabled:opacity-50 text-center"
              />
              <span className="text-text-muted text-xs">episodes</span>
              <button
                onClick={handleEvaluate}
                disabled={isTraining || isEvaluating}
                className="flex-1 flex items-center justify-center gap-2 py-2 rounded-lg font-semibold text-sm text-white bg-gradient-to-br from-accent-yellow to-[#D97706] hover:brightness-115 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <BarChart3 size={16} />
                {isEvaluating ? "Evaluating..." : "Evaluate"}
              </button>
            </div>

            {/* Progress bar during evaluation */}
            {isEvaluating && (
              <div className="mb-3">
                <div className="flex justify-between mb-1">
                  <span className="text-text-secondary text-[11px]">
                    Episode {evalEpisode}/{evalTotalEpisodes}
                  </span>
                  <span className="text-text-secondary text-[11px]">
                    {Math.round((evalEpisode / evalTotalEpisodes) * 100)}%
                  </span>
                </div>
                <div className="w-full h-2 bg-bg-input rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-accent-yellow to-[#D97706] transition-all duration-300"
                    style={{ width: `${(evalEpisode / evalTotalEpisodes) * 100}%` }}
                  />
                </div>
              </div>
            )}

            {/* Evaluation stats */}
            {evalResults.length > 0 && !isEvaluating && (
              <div className="grid grid-cols-2 gap-3 mb-3">
                <div>
                  <span className="text-text-muted text-[10px] uppercase tracking-[0.05em] block">
                    Success Rate
                  </span>
                  <span className="font-mono text-base font-bold text-accent-yellow">
                    {evalSuccessRate.toFixed(1)}%
                  </span>
                </div>
                <div>
                  <span className="text-text-muted text-[10px] uppercase tracking-[0.05em] block">
                    Avg Reward
                  </span>
                  <span className="font-mono text-base font-bold text-accent-yellow">
                    {evalAverageReward.toFixed(1)}
                  </span>
                </div>
                <div>
                  <span className="text-text-muted text-[10px] uppercase tracking-[0.05em] block">
                    Max / Min Reward
                  </span>
                  <span className="font-mono text-sm font-bold text-text-primary">
                    {evalMaxReward.toFixed(1)}
                    <span className="text-text-muted mx-1">/</span>
                    <span className="text-accent-watermelon">{evalMinReward.toFixed(1)}</span>
                  </span>
                </div>
                <div>
                  <span className="text-text-muted text-[10px] uppercase tracking-[0.05em] block">
                    Avg Steps
                  </span>
                  <span className="font-mono text-base font-bold text-accent-purple">
                    {evalAverageSteps.toFixed(1)}
                  </span>
                </div>
              </div>
            )}

            {/* Per-episode results table */}
            {evalResults.length > 0 && (
              <div className="max-h-40 overflow-y-auto">
                <table className="w-full text-left">
                  <thead>
                    <tr className="border-b border-[rgba(139,92,246,0.1)]">
                      <th className="text-text-muted text-[10px] font-semibold uppercase tracking-[0.05em] py-1.5 pr-2">
                        Ep
                      </th>
                      <th className="text-text-muted text-[10px] font-semibold uppercase tracking-[0.05em] py-1.5 pr-2 text-right">
                        Reward
                      </th>
                      <th className="text-text-muted text-[10px] font-semibold uppercase tracking-[0.05em] py-1.5 text-right">
                        Steps
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {evalResults.map((r) => (
                      <tr
                        key={r.episode}
                        className="border-b border-[rgba(139,92,246,0.05)] last:border-b-0"
                      >
                        <td className="text-text-secondary text-[11px] font-mono py-1 pr-2">
                          {r.episode}
                        </td>
                        <td className="text-text-primary text-[11px] font-mono py-1 pr-2 text-right">
                          {r.reward.toFixed(1)}
                        </td>
                        <td className="text-text-secondary text-[11px] font-mono py-1 text-right">
                          {r.steps}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Charts */}
          <RewardChart data={rewardHistory} />
          <LossChart data={lossHistory} />

          {/* Live Metrics */}
          <div className="bg-bg-elevated rounded-lg p-4 border border-[rgba(139,92,246,0.08)]">
            <h3 className="text-text-primary text-[13px] font-semibold mb-3">
              Training Metrics
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <span className="text-text-muted text-[11px] uppercase tracking-[0.05em] block">
                  Episode
                </span>
                <span className="font-mono text-lg font-bold text-text-primary">
                  {currentEpisode}
                </span>
              </div>
              <div>
                <span className="text-text-muted text-[11px] uppercase tracking-[0.05em] block">
                  Total Reward
                </span>
                <span className="font-mono text-lg font-bold text-accent-yellow">
                  {totalReward}
                </span>
              </div>
              <div>
                <span className="text-text-muted text-[11px] uppercase tracking-[0.05em] block">
                  Loss
                </span>
                <span className="font-mono text-lg font-bold text-accent-watermelon">
                  {currentLoss.toFixed(4)}
                </span>
              </div>
              <div>
                <span className="text-text-muted text-[11px] uppercase tracking-[0.05em] block">
                  Steps
                </span>
                <span className="font-mono text-lg font-bold text-accent-purple">
                  {stepCount}
                </span>
              </div>
            </div>
          </div>

          {/* Legend */}
          <div
            className="rounded-lg p-3 flex flex-wrap gap-4"
            style={{
              backgroundColor: "rgba(10, 10, 15, 0.9)",
              backdropFilter: "blur(8px)",
            }}
          >
            {[
              { color: "#8B5CF6", label: "Agent" },
              { color: "#FACC15", label: "Target / Reward" },
              { color: "#FF6B6B", label: "Penalty / Danger" },
              { color: "#6B7280", label: "Empty" },
            ].map((item) => (
              <div key={item.label} className="flex items-center gap-2">
                <span
                  className="w-1.5 h-1.5 rounded-full"
                  style={{ backgroundColor: item.color }}
                />
                <span className="text-text-secondary text-xs">{item.label}</span>
              </div>
            ))}
          </div>
        </motion.aside>
      </div>

      {/* Help Tooltip */}
      <AnimatePresence>
        {showHelp && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-[160]"
              onClick={() => setShowHelp(false)}
            />
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.2 }}
              className="fixed top-16 right-6 z-[170] bg-bg-elevated border border-[rgba(139,92,246,0.2)] rounded-lg p-4 max-w-[320px] shadow-xl"
            >
              {/* Arrow */}
              <div
                className="absolute -top-2 right-6 w-0 h-0"
                style={{
                  borderLeft: "8px solid transparent",
                  borderRight: "8px solid transparent",
                  borderBottom: "8px solid #12121A",
                }}
              />
              <h3 className="text-text-primary text-sm font-semibold mb-3">
                How to Use RL Arena
              </h3>
              <div className="space-y-2.5">
                {[
                  {
                    title: "1. Choose a Game",
                    desc: "Select from Grid World, Cart Pole, or Snake. Each game tests different RL skills.",
                  },
                  {
                    title: "2. Pick an Algorithm",
                    desc: "Choose from Classic (Q-Learning, SARSA), Deep RL (DQN, A3C), or Modern (PPO, TRPO) methods.",
                  },
                  {
                    title: "3. Tune Parameters",
                    desc: "Adjust sliders to see how hyperparameters affect learning speed and stability.",
                  },
                  {
                    title: "4. Watch It Learn",
                    desc: "Hit Start Training and observe the agent improve episode by episode.",
                  },
                ].map((step) => (
                  <div key={step.title}>
                    <span className="text-text-primary text-xs font-medium block">
                      {step.title}
                    </span>
                    <span className="text-text-secondary text-xs leading-relaxed">
                      {step.desc}
                    </span>
                  </div>
                ))}
              </div>
              <p className="text-accent-yellow text-xs italic mt-3">
                Tip: Try the same game with different algorithms to compare their
                learning curves!
              </p>
            </motion.div>
          </>
        )}
      </AnimatePresence>

      {/* Algorithm Detail Modal */}
      <AnimatePresence>
        {showModal && modalAlgorithm && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-[150] flex items-center justify-center p-4"
            style={{
              backgroundColor: "rgba(5, 5, 5, 0.85)",
              backdropFilter: "blur(8px)",
            }}
            onClick={() => setShowModal(false)}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.3, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
              className="bg-bg-elevated rounded-xl border border-[rgba(139,92,246,0.2)] max-w-[640px] max-h-[80vh] w-full overflow-y-auto p-6 relative"
              style={{
                boxShadow: "0 0 60px rgba(139, 92, 246, 0.15)",
              }}
              onClick={(e) => e.stopPropagation()}
            >
              {/* Close button */}
              <button
                onClick={() => setShowModal(false)}
                className="absolute top-4 right-4 text-text-muted hover:text-text-primary transition-colors"
              >
                <X size={20} />
              </button>

              {/* Category Badge */}
              <span
                className={`inline-block text-[11px] font-semibold px-2.5 py-0.5 rounded-full border ${
                  categoryBadgeColors[modalAlgorithm.category] ?? ""
                }`}
              >
                {modalAlgorithm.category}
              </span>

              {/* Algorithm Name */}
              <h2 className="text-text-primary text-2xl font-bold mt-3">
                {modalAlgorithm.name}
              </h2>

              {/* Full Name + Year */}
              <div className="flex items-center gap-2 mt-1">
                <span className="text-text-secondary text-sm">
                  {modalAlgorithm.fullName}
                </span>
                <span className="text-text-muted text-[11px] bg-bg-input px-2 py-0.5 rounded">
                  {modalAlgorithm.year}
                </span>
              </div>

              {/* Divider */}
              <div className="my-4 h-px bg-[rgba(139,92,246,0.15)]" />

              {/* Body Text */}
              <p className="text-text-secondary text-sm leading-[1.7] whitespace-pre-line">
                {modalAlgorithm.detailText}
              </p>

              {/* Key Characteristics */}
              <h3 className="text-text-primary text-sm font-semibold mt-6 mb-2">
                Key Characteristics
              </h3>
              <ul className="space-y-1.5">
                {getCharacteristics(modalAlgorithm.id).map((item) => (
                  <li key={item} className="flex items-start gap-2">
                    <span className="text-accent-purple mt-1">&#8226;</span>
                    <span className="text-text-secondary text-sm">{item}</span>
                  </li>
                ))}
              </ul>

              {/* Supported Games */}
              <h3 className="text-text-primary text-sm font-semibold mt-6 mb-2">
                Supported Games
              </h3>
              <div className="flex gap-2">
                {modalAlgorithm.supportedGames.map((gameId) => {
                  const gameNames: Record<string, string> = {
                    "grid-world": "Grid World",
                    "cart-pole": "Cart Pole",
                    snake: "Snake",
                  };
                  return (
                    <span
                      key={gameId}
                      className="text-[11px] font-semibold px-2.5 py-0.5 rounded-full border text-accent-purple bg-[rgba(139,92,246,0.15)] border-[rgba(139,92,246,0.3)]"
                    >
                      {gameNames[gameId] ?? gameId}
                    </span>
                  );
                })}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function getCharacteristics(algoId: string): string[] {
  const chars: Record<string, string[]> = {
    "q-learning": [
      "Off-policy learning",
      "Epsilon-greedy exploration",
      "Tabular value storage",
      "Guaranteed convergence for finite MDPs",
    ],
    sarsa: [
      "On-policy learning",
      "Conservative exploration",
      "Tabular value storage",
      "Safer in risky environments",
    ],
    "dqn-grid": [
      "Deep neural network function approximation",
      "Experience replay for sample efficiency",
      "Target network for stable learning",
      "Breakthrough for high-dimensional inputs",
    ],
    "a3c-grid": [
      "Parallel actor-learners",
      "Combined policy and value gradients",
      "No replay buffer needed",
      "Asynchronous updates",
    ],
    reinforce: [
      "Monte Carlo policy gradient",
      "Direct policy optimization",
      "Unbiased but high variance",
      "Foundation for modern policy methods",
    ],
    "actor-critic": [
      "Combined value + policy estimation",
      "Lower variance than REINFORCE",
      "Bootstrapped advantage estimates",
      "Bridge to modern deep RL",
    ],
    ppo: [
      "On-policy training",
      "Clipped surrogate objective",
      "Easy to implement vs TRPO",
      "Industry standard for LLM fine-tuning",
    ],
    trpo: [
      "Trust region constraint",
      "Monotonic improvement guarantee",
      "Natural gradient descent",
      "Second-order optimization",
    ],
    "dqn-snake": [
      "Deep neural network Q-learning",
      "Experience replay buffer",
      "Spatial reasoning for grid navigation",
      "Target network stabilization",
    ],
    "a3c-snake": [
      "Parallel exploration of strategies",
      "Handles growing snake complexity",
      "Shared global parameters",
      "Robust to diverse play styles",
    ],
  };
  return chars[algoId] ?? ["Reinforcement Learning Algorithm"];
}
