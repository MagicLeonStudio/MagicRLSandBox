# RL Sandbox Tiny

<p align="center">
  <img src="https://img.shields.io/badge/React-19-61DAFB?logo=react" alt="React 19">
  <img src="https://img.shields.io/badge/TypeScript-5.6-3178C6?logo=typescript" alt="TypeScript">
  <img src="https://img.shields.io/badge/Tailwind-3.4-06B6D4?logo=tailwindcss" alt="Tailwind CSS">
  <img src="https://img.shields.io/badge/Vite-7.2-646CFF?logo=vite" alt="Vite">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/version-v0.9-purple" alt="Version">
</p>

An interactive, browser-based reinforcement learning (RL) demonstration platform. From classic Q-Learning (1989) to modern PPO (2017) — watch AI agents learn in real-time through three sandbox games.

**[Live Demo](https://n4lfyxzsxcx76.ok.kimi.link)**

## Features

- **3 Sandbox Games**
  - **Grid World** (5x5) — Navigate from start to target while avoiding obstacles
  - **Cart Pole** — Balance a pole on a moving cart using physics simulation
  - **Snake** (10x10) — Classic snake game with compact 11-dim state features

- **10 RL Algorithms** spanning 3 eras
  - **Classic (1989-1994):** Q-Learning, SARSA, REINFORCE
  - **Deep RL (2000-2016):** DQN, Actor-Critic, A3C
  - **Modern (2015-2017):** TRPO, **PPO** (the core of LLM RLHF)

- **Real-Time Training Visualization**
  - Pixel-art canvas rendering with dark terminal aesthetic
  - Live reward/loss curves with 20-episode moving average
  - Dynamic hyperparameter sliders (adjustable during training)
  - Training metrics panel (episodes, steps, epsilon decay)

- **Evaluation Mode**
  - Test trained models with greedy policy (epsilon = 0)
  - Configurable episode count (proportional to training episodes)
  - Slow-motion rendering (200ms/step) for behavior observation
  - Statistics: success rate, avg/max/min reward, avg steps

- **Algorithm Details**
  - Each algorithm has a dedicated info card with description
  - "Read More" modal with full technical explanation
  - Parameter configuration per algorithm

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | React 19 + TypeScript |
| Bundler | Vite 7.2 |
| Styling | Tailwind CSS 3.4 + shadcn/ui |
| Animation | Framer Motion |
| Charts | Recharts |
| Icons | Lucide React |
| RL Engine | Custom implementations (pure JS/TS) |

All RL algorithms are implemented from scratch — no TensorFlow.js or external ML libraries required.

## Architecture

```
src/
  algorithms/          # RL algorithm implementations
    NeuralNetwork.ts   # Lightweight NN (ReLU, forward/backprop)
    QLearning.ts       # Tabular Q-Learning
    SARSA.ts           # Tabular SARSA (on-policy)
    DQN.ts             # Deep Q-Network (experience replay, target network)
    REINFORCE.ts       # Monte Carlo policy gradient
    ActorCritic.ts     # Advantage Actor-Critic
    A3C.ts             # Async Advantage Actor-Critic
    PPO.ts             # Proximal Policy Optimization (clipped objective)
    TRPO.ts            # Trust Region Policy Optimization
  engines/             # Game environments
    GridWorldEngine.ts # 5x5 grid navigation
    CartPoleEngine.ts  # Physics-based pole balancing
    SnakeEngine.ts     # 10x10 snake with 11-dim features
  data/
    algorithms.ts      # Algorithm metadata & parameters
  pages/
    Playground.tsx     # Main interactive page
  components/
    Navbar.tsx         # Header with game/algorithm selectors
    Layout.tsx         # Page layout wrapper
    RewardChart.tsx    # Real-time reward curve (raw + MA20)
    LossChart.tsx      # Real-time loss curve (raw + MA20)
```

## Quick Start

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# Production build
npm run build
```

## Color Palette

| Token | Hex | Usage |
|-------|-----|-------|
| Background | `#050505` | Primary background |
| Purple | `#8B5CF6` | Agent, active states, accent |
| Yellow | `#FACC15` | Positive reward, highlights |
| Watermelon Red | `#FF6B6B` | Negative reward, danger |

## Algorithm Coverage

| Algorithm | Year | Category | Games | Key Feature |
|-----------|------|----------|-------|-------------|
| Q-Learning | 1989 | Classic Value | Grid World | Off-policy TD, epsilon-greedy |
| SARSA | 1994 | Classic Value | Grid World | On-policy TD |
| REINFORCE | 1992 | Policy Gradient | Cart Pole | Monte Carlo, normalized returns |
| DQN | 2013 | Deep Value | Grid World, Snake | Experience replay, target network |
| Actor-Critic | 2000 | Actor-Critic | Cart Pole | Shared network, entropy bonus |
| A3C | 2016 | Actor-Critic | Grid World, Snake | Async parallel (simulated) |
| TRPO | 2015 | Policy Gradient | Cart Pole | Trust region constraint |
| PPO | 2017 | Policy Gradient | Cart Pole, Snake | Clipped surrogate objective |

## Changelog

### v0.9 — Companion to Column Article 4 (MC vs TD)
- New MC vs TD demo (`/tutorials/mc-td`): identical trajectories fed to first-visit MC, TD(0) and TD(lambda); dual value-map comparison on a unified color scale; V(s0) learning curve with DP reference line; gamma / alpha / lambda sliders
- New module `src/dp/MCTD.ts`: seeded RNG for reproducible trajectories, first-visit MC, TD(0), TD(lambda) with accumulating eligibility traces, exact policy evaluation as reference
- `ValueHeatmap`: optional vmin/vmax props for comparable color scales across maps

### v0.8 — Companion to Column Articles 2–3
- Routing: new Home page (algorithm matrix), tutorial system at `/tutorials`, playground moved to `/playground`
- New DP demo: `GridWorldModel` + `ValueIteration` + `PolicyIteration` (`src/dp/`)
- New components: `ValueHeatmap` (V-value heatmap + policy arrows), `DPDemo` (sweep animation), `TeX` (KaTeX formulas)
- Fix: GitHub baseline synced to the full v0.7 feature set
- Fix: sub-route asset 404 under `base: './'` (switched to HashRouter)
- Fix: removed Google Fonts dependency and the `plugin-inspect-react-code` debug plugin
- Engineering: added `WORKFLOW.md` (column × sandbox workflow conventions)

### v0.7
- Widened sidebar to 520px for better parameter control
- Fixed PPO numerical stability (gradient clipping, NaN guards)
- Fixed A3C/ActorCritic shared network gradient computation
- Fixed evaluation mode dead loop (max steps limit)
- Improved Snake state representation (100-dim → 11-dim features)
- Added moving average (MA20) to reward/loss charts
- Added evaluation mode with greedy policy testing

### v0.6
- Rewrote all policy gradient algorithms with correct math
- Fixed MSE backward() misuse in REINFORCE/A3C/ActorCritic/PPO/TRPO
- Added proper `backwardOutputGradient()` to NeuralNetwork
- Fixed DQN batch update logic

### v0.5
- Moved HUD above game canvas (no longer overlaps)
- Reduced button font size
- Initial release with 3 games × 8 algorithms

## License

MIT

## Acknowledgments

Built for learners exploring the evolution of reinforcement learning — from tabular methods to the algorithms powering modern large language models.
