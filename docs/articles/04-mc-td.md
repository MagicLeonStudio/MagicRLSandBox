# 「强化学习简史 · 04」MC 与 TD：不依赖模型的两种人生

> Zhihu column: [「强化学习简史」](https://www.zhihu.com/column/c_2087560950404728290) | Interactive demo: [MagicRLSandBox tutorials](https://github.com/MagicLeonStudio/MagicRLSandBox#interactive-tutorialscolumn-companions)

## Abstract

Column article 04. Model-free policy evaluation: Monte Carlo (unbiased, high-variance, episodic, offline) vs Temporal Difference (bootstrapped, low-variance, online). TD(lambda) and eligibility traces as the continuous spectrum between them. Companion to the `mc-td` tutorial page in MagicRLSandBox (v0.9).

## Sections

1. Model confiscated: from Bellman's exact solver back to Ulam's casino (Monte Carlo origin, Los Alamos 1946).
2. MC: the person who waits for the ending — first-visit updates with actual returns G_t.
3. TD: the person who learns step by step — TD error, bootstrapping, Sutton (1988); the streaming-vs-batch analogy.
4. TD(lambda): eligibility traces, the credit-assignment spectrum (forward/backward views in a collapsible).
5. Sandbox hands-on: three estimators fed identical trajectories; dual value maps; V(s0) learning curve vs DP reference line; gamma/alpha/lambda sliders.

## Key formulas

$$
V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)], \qquad
\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
$$

$$
e_t(s) = \gamma\lambda e_{t-1}(s) + \mathbb{1}(S_t = s), \qquad
V(s) \leftarrow V(s) + \alpha \delta_t e_t(s)
$$

## Reading

1. Metropolis (1987). *The Beginning of the Monte Carlo Method*, Los Alamos Science.
2. Sutton (1988). *Learning to Predict by the Methods of Temporal Differences*, Machine Learning 3(1).
3. Barto, Sutton & Anderson (1983). *Neuronlike Adaptive Elements...*, IEEE TSMC.
4. Sutton & Barto (2018), ch. 6-7.

## Companion demo

- Tutorial page: `/tutorials/mc-td` in MagicRLSandBox (v0.9) — identical seeded trajectories fed to first-visit MC, TD(0) and TD(lambda); dual value maps on a unified color scale; V(s0) learning curve with DP reference line; gamma / alpha / lambda sliders.
