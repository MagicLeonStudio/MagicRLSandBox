/** 版本节奏 —— 与 WORKFLOW.md 的版本表保持一致，主页「版本节奏」区块使用 */
export interface Release {
  version: string;
  scope: string; // 配套篇目
  status: "released" | "current" | "planned";
}

export const releases: Release[] = [
  { version: "v0.7", scope: "训练场基础（8 算法 × 3 环境）", status: "released" },
  { version: "v0.8", scope: "第 2–3 篇（MDP、Bellman 与 DP）", status: "released" },
  { version: "v0.9", scope: "第 4–5 篇（MC/TD、Q-learning/SARSA 悬崖行走）", status: "current" },
  { version: "v1.0", scope: "第 6–8 篇（策略梯度、TRPO/PPO、DQN）", status: "planned" },
  { version: "v1.1", scope: "第 9–11 篇（AlphaGo、模型基、探索）", status: "planned" },
  { version: "v1.2", scope: "第 12–15 篇（RLHF 时代）", status: "planned" },
  { version: "v1.3", scope: "第 16–18 篇（推理与 Agent RL）", status: "planned" },
  { version: "v1.4", scope: "第 19–20 篇（总回顾时间线）", status: "planned" },
];
