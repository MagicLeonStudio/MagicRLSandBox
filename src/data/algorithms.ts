export interface AlgorithmParam {
  name: string;
  min: number;
  max: number;
  default: number;
  step: number;
}

export interface Algorithm {
  id: string;
  name: string;
  fullName: string;
  category: "Classic" | "Deep RL" | "Modern";
  year: number;
  description: string;
  detailText: string;
  parameters: AlgorithmParam[];
  supportedGames: string[];
}

export const algorithms: Algorithm[] = [
  // Grid World algorithms
  {
    id: "q-learning",
    name: "Q-Learning",
    fullName: "Q-Learning",
    category: "Classic",
    year: 1989,
    description:
      "The foundational off-policy RL algorithm. Q-Learning learns action-values by bootstrapping from the maximum estimated future reward, making it simple yet powerful for discrete state spaces.",
    detailText:
      "Q-Learning is one of the most important algorithms in reinforcement learning history. It was one of the first algorithms to guarantee convergence to the optimal policy for finite Markov Decision Processes. The key insight is the Bellman equation — the optimal action-value function can be learned by iteratively updating estimates based on observed rewards and maximum future values. Q-Learning is off-policy, meaning it can learn about the optimal policy while still exploring with a different behavior policy. This makes it more sample-efficient than on-policy alternatives like SARSA. Today, Q-Learning serves as the conceptual foundation for deep RL methods like DQN.",
    parameters: [
      { name: "learningRate", min: 0.001, max: 0.5, default: 0.1, step: 0.001 },
      { name: "discount", min: 0.5, max: 0.99, default: 0.95, step: 0.01 },
      { name: "epsilon", min: 0.01, max: 1.0, default: 0.3, step: 0.01 },
      { name: "epsilonDecay", min: 0.99, max: 0.9999, default: 0.998, step: 0.0001 },
    ],
    supportedGames: ["grid-world"],
  },
  {
    id: "sarsa",
    name: "SARSA",
    fullName: "State-Action-Reward-State-Action",
    category: "Classic",
    year: 1994,
    description:
      "An on-policy temporal difference algorithm. Unlike Q-Learning, SARSA updates its estimates using the actual next action taken, making it more conservative and often safer in practice.",
    detailText:
      "SARSA gets its name from the sequence of events that define each update: State, Action, Reward, next State, next Action. Unlike Q-Learning which always assumes the best future action, SARSA uses whichever action was actually selected — including exploratory random moves. This makes SARSA on-policy: it learns about the policy it is currently following, including exploration. The result is that SARSA tends to learn safer, more conservative policies. In environments where taking a bad action has severe consequences, SARSA often outperforms Q-Learning. The trade-off is that SARSA can be slower to find optimal paths since it accounts for its own exploration noise during learning.",
    parameters: [
      { name: "learningRate", min: 0.001, max: 0.5, default: 0.1, step: 0.001 },
      { name: "discount", min: 0.5, max: 0.99, default: 0.95, step: 0.01 },
      { name: "epsilon", min: 0.01, max: 1.0, default: 0.3, step: 0.01 },
      { name: "epsilonDecay", min: 0.99, max: 0.9999, default: 0.998, step: 0.0001 },
    ],
    supportedGames: ["grid-world"],
  },
  {
    id: "dqn-grid",
    name: "DQN",
    fullName: "Deep Q-Network",
    category: "Deep RL",
    year: 2013,
    description:
      "DeepMind's breakthrough algorithm that combined Q-Learning with deep neural networks. Uses experience replay and target networks to stabilize learning in complex environments.",
    detailText:
      "DQN revolutionized reinforcement learning by proving that neural networks could learn effective policies directly from high-dimensional sensory input. Two key innovations made this possible: experience replay, which stores and randomly samples past experiences to break correlation in training data, and target networks, which use a slowly-updating separate network to compute target values for stability. DQN demonstrated superhuman performance on Atari games, learning to play directly from pixel inputs. The algorithm spawned an entire family of improvements including Double DQN, Dueling DQN, and Prioritized Experience Replay. DQN showed that the fundamental Q-Learning approach scales to deep neural networks.",
    parameters: [
      { name: "learningRate", min: 0.0001, max: 0.01, default: 0.001, step: 0.0001 },
      { name: "epsilon", min: 0.01, max: 1.0, default: 1.0, step: 0.01 },
      { name: "batchSize", min: 16, max: 128, default: 32, step: 1 },
      { name: "replayBuffer", min: 1000, max: 10000, default: 5000, step: 100 },
      { name: "targetUpdate", min: 10, max: 500, default: 100, step: 10 },
    ],
    supportedGames: ["grid-world"],
  },
  {
    id: "a3c-grid",
    name: "A3C",
    fullName: "Asynchronous Advantage Actor-Critic",
    category: "Deep RL",
    year: 2016,
    description:
      "DeepMind's algorithm that uses multiple parallel agents to learn more efficiently. Combines policy gradients with value function estimation for stable learning.",
    detailText:
      "A3C was a breakthrough in deep reinforcement learning that showed how parallel actor-learners could stabilize and accelerate training. Multiple agents explore the environment simultaneously, each with their own policy and value network parameters. Updates are applied asynchronously to shared global parameters. The key insight is that parallel exploration naturally decorrelates experiences — different agents encounter different states at different times. A3C combines the actor-critic architecture: the actor (policy network) decides which actions to take, while the critic (value network) estimates how good those actions are. The advantage function — the difference between actual returns and expected values — provides a low-variance signal for policy improvement. A3C achieved state-of-the-art results on Atari games while training faster than DQN.",
    parameters: [
      { name: "learningRate", min: 0.0001, max: 0.01, default: 0.001, step: 0.0001 },
      { name: "gamma", min: 0.9, max: 0.99, default: 0.99, step: 0.01 },
      { name: "entropyCoef", min: 0.001, max: 0.1, default: 0.01, step: 0.001 },
      { name: "valueCoef", min: 0.1, max: 1.0, default: 0.5, step: 0.1 },
    ],
    supportedGames: ["grid-world"],
  },
  // Cart Pole algorithms
  {
    id: "reinforce",
    name: "REINFORCE",
    fullName: "Monte Carlo Policy Gradient",
    category: "Classic",
    year: 1992,
    description:
      "The foundational policy gradient algorithm. REINFORCE directly optimizes the policy by following the gradient of expected cumulative reward, using Monte Carlo returns for unbiased estimates.",
    detailText:
      "REINFORCE is the grandfather of all policy gradient methods. Instead of learning value functions and deriving policies from them (like Q-Learning), REINFORCE directly parameterizes and optimizes the policy itself. The key insight is the policy gradient theorem: the gradient of expected reward can be estimated from sampled trajectories without knowing the environment dynamics. REINFORCE uses Monte Carlo returns — the actual cumulative reward from each state to the end of the episode — as its learning signal. This makes it unbiased but high-variance. The algorithm paved the way for actor-critic methods, which combine policy gradients with value function bootstrapping to reduce variance. REINFORCE is conceptually elegant and forms the theoretical foundation for modern policy optimization.",
    parameters: [
      { name: "learningRate", min: 0.0001, max: 0.01, default: 0.001, step: 0.0001 },
      { name: "gamma", min: 0.9, max: 0.99, default: 0.99, step: 0.01 },
    ],
    supportedGames: ["cart-pole"],
  },
  {
    id: "actor-critic",
    name: "Actor-Critic",
    fullName: "Advantage Actor-Critic",
    category: "Deep RL",
    year: 2000,
    description:
      "Combines value-based and policy-based methods. The actor learns a policy while the critic evaluates it, providing lower-variance gradient estimates than pure policy gradients.",
    detailText:
      "Actor-Critic methods combine the best of both worlds: the direct policy optimization of REINFORCE with the variance reduction of value function bootstrapping. The actor is a policy network that decides which actions to take, while the critic is a value network that estimates how good the current state is. The critic's value estimates are used to compute an advantage function — how much better an action was than expected. This advantage signal has much lower variance than raw Monte Carlo returns, enabling more stable learning. Actor-Critic bridges classic temporal difference methods with modern deep policy optimization. Many of today's most successful RL algorithms, including PPO and A3C, are descendants of the Actor-Critic framework.",
    parameters: [
      { name: "learningRate", min: 0.0001, max: 0.01, default: 0.001, step: 0.0001 },
      { name: "gamma", min: 0.9, max: 0.99, default: 0.99, step: 0.01 },
      { name: "entropyCoef", min: 0.001, max: 0.1, default: 0.01, step: 0.001 },
    ],
    supportedGames: ["cart-pole"],
  },
  {
    id: "ppo",
    name: "PPO",
    fullName: "Proximal Policy Optimization",
    category: "Modern",
    year: 2017,
    description:
      "A modern policy gradient method that uses clipped surrogate objectives to prevent overly large policy updates. It's stable, sample-efficient, and powers RLHF in large language models.",
    detailText:
      "PPO is a policy gradient method that simplified and stabilized the earlier TRPO algorithm. Instead of complex trust region constraints, PPO uses a simple clipped objective that prevents the policy from changing too much in a single update. Think of it as 'baby steps' for the AI — each update is small and safe, but over thousands of iterations, the policy dramatically improves. PPO's simplicity, stability, and strong performance made it the go-to algorithm for a wide range of RL tasks. Most notably, PPO became the standard algorithm for RLHF (Reinforcement Learning from Human Feedback) that powers ChatGPT, Claude, and other large language models. Its balance of implementation simplicity and empirical performance makes it the most widely used modern RL algorithm.",
    parameters: [
      { name: "learningRate", min: 0.0001, max: 0.01, default: 0.0003, step: 0.0001 },
      { name: "gamma", min: 0.9, max: 0.99, default: 0.99, step: 0.01 },
      { name: "clipEpsilon", min: 0.1, max: 0.3, default: 0.2, step: 0.01 },
      { name: "entropyCoef", min: 0.001, max: 0.1, default: 0.01, step: 0.001 },
    ],
    supportedGames: ["cart-pole", "snake"],
  },
  {
    id: "trpo",
    name: "TRPO",
    fullName: "Trust Region Policy Optimization",
    category: "Modern",
    year: 2015,
    description:
      "Uses trust region constraints to ensure stable policy updates. Each update is bounded by a KL divergence constraint, preventing destructive large steps.",
    detailText:
      "TRPO addressed a critical problem in policy gradient methods: overly aggressive updates could collapse policy performance irreversibly. TRPO's solution was a trust region constraint — each policy update must keep the new policy within a bounded KL divergence from the old one. This is like saying 'you can change, but not too much at once.' The constraint is enforced via a complex second-order optimization using conjugate gradient and line search. TRPO proved that monotonic policy improvement guarantees from theory could be approximated in practice. While its implementation complexity led to PPO's development, TRPO's theoretical contributions — connecting policy gradients to natural gradient descent — were profoundly influential. TRPO showed that constrained optimization was key to stable policy learning.",
    parameters: [
      { name: "learningRate", min: 0.0001, max: 0.01, default: 0.001, step: 0.0001 },
      { name: "gamma", min: 0.9, max: 0.99, default: 0.99, step: 0.01 },
      { name: "maxKL", min: 0.001, max: 0.02, default: 0.01, step: 0.001 },
    ],
    supportedGames: ["cart-pole"],
  },
  // Snake algorithms
  {
    id: "dqn-snake",
    name: "DQN",
    fullName: "Deep Q-Network",
    category: "Deep RL",
    year: 2013,
    description:
      "DeepMind's breakthrough algorithm applied to Snake. Uses a neural network to approximate Q-values and experience replay for stable learning in a grid-based environment.",
    detailText:
      "DQN revolutionized reinforcement learning by proving that neural networks could learn effective policies directly from high-dimensional sensory input. Two key innovations made this possible: experience replay, which stores and randomly samples past experiences to break correlation in training data, and target networks, which use a slowly-updating separate network to compute target values for stability. When applied to Snake, DQN must learn spatial reasoning — understanding where the food is relative to the snake's head, avoiding walls and self-collisions, and planning growth. The neural network processes the game state (often the full grid) and outputs Q-values for each direction. DQN demonstrated that deep RL could handle complex spatial navigation tasks.",
    parameters: [
      { name: "learningRate", min: 0.0001, max: 0.01, default: 0.001, step: 0.0001 },
      { name: "epsilon", min: 0.01, max: 1.0, default: 1.0, step: 0.01 },
      { name: "batchSize", min: 16, max: 128, default: 32, step: 1 },
      { name: "replayBuffer", min: 1000, max: 10000, default: 5000, step: 100 },
      { name: "targetUpdate", min: 10, max: 500, default: 100, step: 10 },
    ],
    supportedGames: ["snake"],
  },
  {
    id: "a3c-snake",
    name: "A3C",
    fullName: "Asynchronous Advantage Actor-Critic",
    category: "Deep RL",
    year: 2016,
    description:
      "Applies parallel actor-learners to Snake. Multiple agents explore different strategies simultaneously, learning robust navigation and food-seeking behavior.",
    detailText:
      "A3C brings the power of parallel learning to the Snake environment. Multiple actor-learners explore the game simultaneously, each following slightly different policies. This natural exploration diversity helps the shared network discover robust strategies for food collection and collision avoidance. In Snake specifically, A3C must learn the challenge of growing — early in training, a short snake easily navigates to food, but as it grows longer, self-collision becomes a major risk. The advantage-based learning signal helps the agent understand which moves lead to long-term survival versus short-term food rewards. A3C's parallel architecture is particularly effective for Snake, where different agents can explore different play styles — conservative wall-hugging versus aggressive center play.",
    parameters: [
      { name: "learningRate", min: 0.0001, max: 0.01, default: 0.001, step: 0.0001 },
      { name: "gamma", min: 0.9, max: 0.99, default: 0.99, step: 0.01 },
      { name: "entropyCoef", min: 0.001, max: 0.1, default: 0.01, step: 0.001 },
      { name: "valueCoef", min: 0.1, max: 1.0, default: 0.5, step: 0.1 },
    ],
    supportedGames: ["snake"],
  },
];

export const games = [
  { id: "grid-world", name: "Grid World" },
  { id: "cart-pole", name: "Cart Pole" },
  { id: "snake", name: "Snake" },
];

export function getAlgorithmsForGame(gameId: string): Algorithm[] {
  return algorithms.filter((a) => a.supportedGames.includes(gameId));
}

export function getAlgorithmById(id: string): Algorithm | undefined {
  return algorithms.find((a) => a.id === id);
}
