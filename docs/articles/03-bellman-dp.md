# 「强化学习简史 · 03」Bellman 方程与动态规划：一切的源头，一切的诅咒

> Zhihu column: [「强化学习简史」](https://www.zhihu.com/column/c_2087560950404728290) | Interactive demo: [MagicRLSandBox tutorials](https://github.com/MagicLeonStudio/MagicRLSandBox#interactive-tutorialscolumn-companions)

## Abstract

Column article 03. The Bellman optimality equation as a fixed-point equation: the principle of optimality, Value Iteration vs Policy Iteration, contraction-mapping convergence with a loss-bounded stopping criterion, and the curse of dimensionality as the engine of 60 years of RL history. Companion to the `bellman-dp` tutorial page in MagicRLSandBox (v0.8).

## Sections

1. Principle of Optimality: the self-similarity observation that turns global search into a local equation.
2. Bellman equation: V^pi as a self-referential recursion; the optimality equation as a fixed-point problem.
3. VI vs PI — two ways to approach the same fixed point; gamma-contraction, Banach theorem, and the loss-bounded stopping criterion (callout in the article).
4. Curse of dimensionality: why the exact solver is unreachable in perceptual state spaces, and why that single fact drives sampling methods, function approximation, and deep RL.
5. Sandbox hands-on: step-through sweeps, gamma slider re-pricing, VI/PI comparison (measured: both converge in ~9 iterations at gamma=0.99 on the 5x5 GridWorld).

## Key formulas

$$
V^{\pi}(s) = \sum_{s'} P\big(s' \mid s, \pi(s)\big) \Big[ R(s, a, s') + \gamma V^{\pi}(s') \Big], \qquad
V^{*}(s) = \max_a \sum_{s'} P(s' \mid s, a)\Big[ R(s, a, s') + \gamma V^{*}(s') \Big]
$$

$$
\| TV - TU \|_{\infty} \le \gamma \| V - U \|_{\infty}, \qquad
\| V_k - V^* \|_{\infty} \le \gamma^{k} \| V_0 - V^* \|_{\infty}
$$

Stopping criterion: residual $\|V_{k+1}-V_k\|_\infty < \epsilon(1-\gamma)/(2\gamma) \Rightarrow \|V_{k+1}-V^*\|_\infty < \epsilon$.

## Reading

1. Bellman (1957). *Dynamic Programming*, ch. 1-2.
2. Bellman (1957). *A Markovian Decision Process*, J. Math. Mech.
3. Howard (1960). *Dynamic Programming and Markov Processes* (MIT Press).
4. Sutton & Barto (2018), ch. 4.

## Companion demo

- Tutorial page: `/tutorials/bellman-dp` in MagicRLSandBox (v0.8) — VI/PI step-through demo with V-value heatmap, greedy policy arrows, Bellman residual curve (log scale), gamma slider, and optimal-path highlight.
