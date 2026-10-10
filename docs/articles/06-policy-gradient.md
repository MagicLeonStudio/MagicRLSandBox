# 「强化学习简史 · 06」策略梯度定理：当 RL 学会「直接优化」

> Zhihu column: [「强化学习简史」](https://www.zhihu.com/column/c_2087560950404728290) | Interactive demo: [MagicRLSandBox tutorials](https://github.com/MagicLeonStudio/MagicRLSandBox#interactive-tutorialscolumn-companions)

## Abstract

Column article 06. From value-based control (argmax, non-differentiable, deterministic-only) to direct policy optimization: the Policy Gradient Theorem (Sutton et al. 2000), REINFORCE (Williams 1992) with its log-prob x return update, the baseline variance-reduction trick, and the path to Actor-Critic. Proof sketch in Appendix 6.1 (score function trick). Companion to the `policy-gradient` tutorial page in MagicRLSandBox.

## Sections

1. Bypassing argmax: continuous optimization over theta; stochastic optimal policies (rock-paper-scissors).
2. Policy Gradient Theorem: nabla J = E[nabla log pi * Q]; reading; score function trick.
3. REINFORCE: G as unbiased Q-sample; update anatomy (one-hot minus softmax probs; return weighting); high variance; baseline (E[nabla log pi * b] = 0).
4. From baseline to Actor-Critic: b -> V^pi(s_t), advantage; bootstrap trades bias for variance; critic genealogy (A2C/A3C/TRPO/PPO/RLHF-RM).
5. Sandbox hands-on: PolicyGradientDemo (policy map evolution, baseline curves, per-step decomposition), experiments, two questions.

## Key formulas

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\big[\nabla \log \pi_\theta(a|s)\, Q^{\pi}(s,a)\big], \qquad
\theta \leftarrow \theta + \alpha\, \nabla \log \pi_\theta(A_t|S_t)\,(G_t - b)
$$

## Reading

1. Williams (1992). *Simple Statistical Gradient-Following Algorithms...*, Machine Learning 8(3-4).
2. Sutton, McAllester, Singh & Mansour (2000). *Policy Gradient Methods...*, NeurIPS.
3. Konda & Tsitsiklis (2000). *Actor-Critic Algorithms*, NeurIPS.
4. Sutton & Barto (2018), ch. 13.

## Companion demo

- Tutorial page: `/tutorials/policy-gradient` in MagicRLSandBox — softmax policy map evolution on 5x5 GridWorld, matched-seed baseline on/off step curves, per-step nabla log pi * (G - b) decomposition, alpha/speed sliders.
