# 「强化学习简史 · 07」从 TRPO 到 PPO：深度 RL 的工程底座是如何浇筑的

> Zhihu column: [「强化学习简史」](https://www.zhihu.com/column/c_2087560950404728290) | Interactive demo: [MagicRLSandBox tutorials](https://github.com/MagicLeonStudio/MagicRLSandBox#interactive-tutorialscolumn-companions)

## Abstract

Column article 07. Step-size catastrophe in policy gradients; Kakade-Langford performance bound and trust-region idea; TRPO (2015) with KL constraint, conjugate gradient, monotonic improvement; PPO (2017) clip objective with min-pessimistic-branch asymmetry; GAE (2016) lambda interpolation between MC and TD. PPO as the default recipe through RLHF. Kakade-Langford bound sketch in Appendix 7.1.

## Sections

1. Step-size catastrophe: no fixed loss landscape to return to in RL (data distribution follows the policy).
2. Trust region: Kakade-Langford bound (real improvement >= surrogate improvement - C*KL^2); TRPO's elegance vs engineering heaviness (Fisher, conjugate gradient, line search).
3. PPO: clip objective; min's asymmetric protection (A>0 caps at 1+eps, A<0 floors at 1-eps); GAE lambda spectrum; first-order trust region.
4. Why PPO rules RLHF: <50 lines core code, minibatch-SGD compatible, two interpretable hyperparameters; InstructGPT 2022.

## Key formulas

$$
\max_\theta\; \mathbb{E}_t\Big[\frac{\pi_\theta}{\pi_{\theta_{old}}} \hat{A}_t\Big] \;\text{s.t.}\; \bar{D}_{KL} \le \delta, \qquad
L^{CLIP} = \mathbb{E}\big[\min(r_t\hat{A}_t, \mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\hat{A}_t)\big]
$$

## Reading

1. Kakade & Langford (2002). *Approximately Optimal Approximate RL*, ICML.
2. Schulman et al. (2015). *Trust Region Policy Optimization*, ICML.
3. Schulman et al. (2017). *Proximal Policy Optimization Algorithms*, arXiv.
4. Schulman et al. (2016). *High-Dimensional Continuous Control Using GAE*, ICLR.

## Companion demo

- Tutorial page: `/tutorials/ppo` in MagicRLSandBox (v1.0) — clip objective plot with A-sign switch and epsilon slider (asymmetric clipping visualization), GAE lambda spectrum bars on a fixed delta trajectory.
