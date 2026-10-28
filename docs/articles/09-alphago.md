# 「强化学习简史 · 09」AlphaGo 三部曲：从围棋到 AlphaZero 的通用棋类之神

> Zhihu column: [「强化学习简史」](https://www.zhihu.com/column/c_2087560950404728290) | Interactive demo: [MagicRLSandBox tutorials](https://github.com/MagicLeonStudio/MagicRLSandBox#interactive-tutorialscolumn-companions)

## Abstract

Column article 09. After DQN's value estimation, AlphaGo introduces decision-time planning: policy and value networks guide MCTS rather than replace it. AlphaGo (2016) combines human-game supervision, self-play value learning and rollouts; AlphaGo Zero (2017) removes human data and rollouts, distilling MCTS visit distributions and game outcomes into one policy-value network; AlphaZero (2017/2018) generalizes the same self-play loop to chess, shogi and Go. The Tiny AlphaZero demo reproduces the full loop on 4x4 three-in-a-row with a perfect solver used only for evaluation.

## Sections

1. Why value estimation alone is insufficient: from Q(s,a) to lookahead and planning.
2. Three subtractions: human games removed, rollouts removed, game-specific engineering removed.
3. MCTS as a policy-improvement operator: PUCT selection, value-head evaluation, alternating backup, and self-play distillation.
4. Tiny AlphaZero demo: 24 self-play games per iteration, 300 simulations per move, 10 training epochs, 100 iterations; perfect minimax only for evaluation.
5. Measured outcome: raw policy starts at 1 win / 11 losses and reaches 9 wins / 3 losses at iteration 100; MCTS reaches 6 wins / 0 losses; final joint loss 0.627.

## Key formulas

$$
a^* = \arg\max_a \left[ Q(s,a) + c_{\mathrm{puct}} P(s,a) \frac{\sqrt{N(s)}}{1+N(s,a)} \right]
$$

$$
\mathcal{L} = (z-v)^2 - \boldsymbol{\pi}^{\top}\log\mathbf{p} + \lambda\|\theta\|^2
$$

MCTS produces the improved visit distribution π and final result z; the network learns p ≈ π and v ≈ z.

## Reading

1. Silver et al. (2016). *Mastering the game of Go with deep neural networks and tree search*, Nature 529, 484–489.
2. Silver et al. (2017). *Mastering the game of Go without human knowledge*, Nature 550, 354–359.
3. Silver et al. (2018). *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play*, Science 362, 1140–1144.
4. Browne et al. (2012). *A Survey of Monte Carlo Tree Search Methods*, IEEE TCIAIG 4(1), 1–43.
5. Kocsis & Szepesvári (2006). *Bandit based Monte-Carlo Planning*, ECML.

## Companion demo

- Tutorial page: `/tutorials/alphago` in MagicRLSandBox (v1.1) — Tiny AlphaZero with live PUCT action statistics, network-prior vs MCTS-visit heatmaps, self-play loss and independent solver evaluation. Solver is evaluation-only and never supplies training data.
