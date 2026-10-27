# 「强化学习简史 · 08」DQN 一夜封神：经验回放 + 目标网络的配方学

> Zhihu column: [「强化学习简史」](https://www.zhihu.com/column/c_2087560950404728290) | Interactive demo: [MagicRLSandBox tutorials](https://github.com/MagicLeonStudio/MagicRLSandBox#interactive-tutorialscolumn-companions)

## Abstract

Column article 08. The end of the tabular era (Atari pixels, 256^(160x210)); DQN (Mnih 2013/2015) as two existing parts assembled into a deep network: experience replay (breaking sample correlation, back to i.i.d. supervision) + target network (freezing the bootstrap reference). Loss formula with theta-minus. Nature 2015 and the variant wave (Double DQN, Dueling, PER, Rainbow). Stability mechanisms made visible in the pixel demo. Double DQN derivation in Appendix 8.1.

## Sections

1. End of the tabular era: the wall for pixel-level state spaces; TD-Gammon's echo.
2. The recipe: replay (correlation breaking) + target net (frozen reference); DQN loss; callout mapping the three pillars to the Deadly Triad; perception-side intuition (streaming vs batch replay).
3. One-night fame and the variant wave: Nature 2015; Double DQN (max overestimation fix), Dueling, PER, Rainbow; DQN as a pluggable base.
4. Sandbox hands-on: DQNDemo pixel-mechanism lab — real CNN activations, replay buffer and batch filmstrips, fixed-probe online/target trajectories, plus a three-way ablation (full / no-replay / no-target-net) with matched initialization and greedy evaluation.

## Key formulas

$$
L(\theta) = \mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}\Big[\big( r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a) \big)^2\Big]
$$

Double DQN: $r + \gamma\, Q_{\theta^-}(s', \arg\max_{a'} Q_{\theta}(s', a'))$.

## Reading

1. Mnih et al. (2013). *Playing Atari with Deep RL*, workshop paper (replay only).
2. Mnih et al. (2015). *Human-level control through deep RL*, Nature (full recipe).
3. van Hasselt et al. (2016). *Double Q-learning*, AAAI.
4. Wang et al. (2016). *Dueling Networks*, ICML.
5. Schaul et al. (2016). *Prioritized Experience Replay*, ICLR.
6. Hessel et al. (2017). *Rainbow*, AAAI.

## Companion demo

- Tutorial page: `/tutorials/dqn` in MagicRLSandBox (v1.0) — pixel Mini GridWorld with a pure-TypeScript two-conv DQN; live CNN feature maps and action Q-values; replay-buffer and sampled-batch visualization; fixed-probe target-network step plot; three-way ablation and greedy policy evaluation.
