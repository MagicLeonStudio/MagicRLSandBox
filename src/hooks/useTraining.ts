import { useState, useCallback } from "react";

export interface TrainingState {
  isTraining: boolean;
  currentEpisode: number;
  totalReward: number;
  currentLoss: number;
  rewardHistory: number[];
  lossHistory: number[];
  stepCount: number;
  epsilon: number;
}

export interface TrainingActions {
  startTraining: () => void;
  stopTraining: () => void;
  resetTraining: () => void;
  setEpsilon: (epsilon: number) => void;
}

export function useTraining(): TrainingState & TrainingActions {
  const [isTraining, setIsTraining] = useState(false);
  const [currentEpisode, setCurrentEpisode] = useState(0);
  const [totalReward, setTotalReward] = useState(0);
  const [currentLoss, setCurrentLoss] = useState(0);
  const [rewardHistory, setRewardHistory] = useState<number[]>([]);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [stepCount, setStepCount] = useState(0);
  const [epsilon, setEpsilon] = useState(0.3);

  const startTraining = useCallback(() => {
    setIsTraining(true);
  }, []);

  const stopTraining = useCallback(() => {
    setIsTraining(false);
  }, []);

  const resetTraining = useCallback(() => {
    setIsTraining(false);
    setCurrentEpisode(0);
    setTotalReward(0);
    setCurrentLoss(0);
    setRewardHistory([]);
    setLossHistory([]);
    setStepCount(0);
    setEpsilon(0.3);
  }, []);

  return {
    isTraining,
    currentEpisode,
    totalReward,
    currentLoss,
    rewardHistory,
    lossHistory,
    stepCount,
    epsilon,
    startTraining,
    stopTraining,
    resetTraining,
    setEpsilon,
  };
}
