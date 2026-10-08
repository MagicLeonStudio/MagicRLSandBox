# 「强化学习简史 · 05」Q-learning（1989）：一张表格开启的离策略革命

> Zhihu column: [「强化学习简史」](https://www.zhihu.com/column/c_2087560950404728290) | Interactive demo: [MagicRLSandBox tutorials](https://github.com/MagicLeonStudio/MagicRLSandBox#interactive-tutorialscolumn-companions)

## Abstract

Column article 05. From evaluation to control: Watkins' action-value Q(s,a) removes the model from policy improvement; Q-learning as the sampled form of the Bellman optimality equation (sampling replaces the sum, bootstrap replaces the expectation, max survives). SARSA and the Cliff Walking experiment (S&B Example 6.6) as the textbook on-policy vs off-policy comparison. Convergence (Watkins & Dayan 1992) in Appendix 5.1, plus foreshadowing of max overestimation bias and the Deadly Triad. Companion to the `q-learning` tutorial page in MagicRLSandBox (v0.9).

## Sections

1. From evaluation to control: Q(s,a) makes argmax a table lookup — no model needed for policy improvement.
2. Q-learning: one update line, three rivers converge (sum -> sampling, expectation -> TD, max preserved). Off-policy callout; SARSA as the one-character-difference twin.
3. Cliff Walking: Q-learning learns the optimal-but-risky 13-step edge path, SARSA the safe 17-step detour; online performance inverted (-59 vs -31). Defensive-driving analogy.
4. Convergence and its shadows: Watkins & Dayan (1992) with Robbins-Monro conditions (Appendix 5.1); max overestimation bias (Double DQN, article 08) and the Deadly Triad (function approximation x bootstrap x off-policy).
5. Sandbox hands-on: dual greedy-policy maps, online curves on matched seeds, epsilon slider experiments, two questions (cliff penalty reshaping).

## Key formulas

$$
Q^{\pi}(s,a) = \mathbb{E}_{\pi}[G_t \mid S_t=s, A_t=a], \qquad \pi'(s) = \arg\max_a Q^{\pi}(s,a)
$$

$$
Q(S_t,A_t) \leftarrow Q + \alpha\big[R + \gamma \max_{a'} Q(S_{t+1},a') - Q\big], \qquad
Q(S_t,A_t) \leftarrow Q + \alpha\big[R + \gamma Q(S_{t+1},A_{t+1}) - Q\big] \;(\text{SARSA})
$$

## Reading

1. Watkins (1989). *Learning from Delayed Rewards*. PhD thesis, Cambridge.
2. Watkins & Dayan (1992). *Q-learning*, Machine Learning 8(3-4).
3. Rummery & Niranjan (1994). *On-line Q-learning using connectionist systems*. Cambridge TR.
4. Sutton & Barto (2018), ch. 6.

## Companion demo

- Tutorial page: `/tutorials/q-learning` in MagicRLSandBox (v0.9) — dual greedy-policy maps on the 4x12 cliff grid (red cliff row), matched-seed online learning curves, epsilon / alpha / speed sliders.
