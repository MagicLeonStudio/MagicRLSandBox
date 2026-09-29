# 「强化学习简史 · 02」马尔可夫决策过程：强化学习世界的元素周期表

> Zhihu column: [「强化学习简史」](https://www.zhihu.com/column/c_2087560950404728290) | Interactive demo: [MagicRLSandBox tutorials](https://github.com/MagicLeonStudio/MagicRLSandBox#interactive-tutorialscolumn-companions)

## Abstract

Column article 02. The Markov Decision Process (MDP) as the unifying problem language of reinforcement learning: the five-tuple $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$, the Markov property, return / policy / objective, and a hands-on GridWorld walkthrough. Companion to the `mdp` tutorial page in MagicRLSandBox (v0.8).

## Sections

1. From chess to life: sequential decision problems — why the three rivers of article 01 needed a riverbed.
2. The MDP five-tuple, element by element; the discount factor $\gamma$ as "how much the future is worth".
3. The Markov property: the bold history-independence assumption; tic-tac-toe vs. poker; POMDP as life's setting.
4. Return, policy and objective — the watershed between RL (learning from consequences) and supervised learning (copying answers); credit assignment foreshadowed.
5. GridWorld: a minimal MDP you can toggle in the browser (5x5, S / gold star / obstacles, -0.1 step penalty, +10 goal).

Full text (Zhihu edition): see Notion column workspace, article 02 page.

## Key formulas

$$
M = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle, \qquad
P(s' \mid s, a) = \Pr(S_{t+1} = s' \mid S_t = s, A_t = a)
$$

$$
G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}, \qquad
J(\pi) = \mathbb{E}_{\pi}[G_t], \qquad
\pi^* = \arg\max_{\pi} J(\pi)
$$

## Reading

1. Bellman (1957). *A Markovian Decision Process*, J. Math. Mech. — the birth certificate of MDP, ten pages.
2. Bellman (1957). *Dynamic Programming*, ch. 1-2.
3. Sutton & Barto (2018). *Reinforcement Learning: An Introduction*, 2nd ed., ch. 3.
4. Puterman (1994). *Markov Decision Processes* (optional).

## Companion demo

- Tutorial page: `/tutorials/mdp` in MagicRLSandBox (v0.8) — GridWorld with MDP semantics.
- Next article (03): Bellman equation & dynamic programming — the `dp` tutorial with step-through Value/Policy Iteration is already integrated.
