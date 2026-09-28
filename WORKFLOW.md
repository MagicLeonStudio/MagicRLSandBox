# WORKFLOW —— 「强化学习简史」专栏 × MagicRLSandBox 工作流规定

本仓库是知乎专栏「强化学习简史」的配套交互式沙盒。专栏文章与沙盒功能按统一节奏联动开发，本文档规定协作流程。

## 核心原则：Demo 先行

每篇专栏文章发布前，其配套的最小可用 demo 必须先在本仓库可运行。

## 每篇文章的标准流程（SOP）

| 阶段 | 动作 | 产出 |
|---|---|---|
| ① 研究 | 文献调研（原论文 + scholar） | 文献拆解卡片 |
| ② demo 开发 | 实现最小可用 demo（新算法 / 新环境 / 新组件），本地验证 | 可运行的沙盒功能 + 曲线截图/录屏素材 |
| ③ 撰写 | 基于 demo 实测结果撰写专栏文章（公式遵循 peco-equation，配图遵循 magic-theme）；标题前缀规范：「强化学习简史 · NN」（两位序号） | 知乎文定稿 |
| ④ 发布 | Notion 同步（页面内「Notion 文章列表」之下）→ 知乎 markdown → 本仓库（教程数据 + 文档同步） | 三端上线 |
| ⑤ 反馈打磨 | 收集读者反馈，增强 demo / 修订文章 | 小版本迭代 |

### 两条护栏

1. **最小可用 demo**：demo 只需支撑文章核心论点即停手，超出需求的功能等读者反馈再迭代；
2. **降级预案**：demo 确实卡壳时，文章可先以静态示意图 + 伪代码上线，注明"交互演示将在后续版本提供"——例外条款，主力顺序不倒置。

## 版本节奏

沙盒大版本随专栏幕次推进，每个 Release 的 notes 链接对应知乎文章；**版本每增加 0.1 必须打 tag**：

| 版本 | 配套篇目 | 状态 |
|---|---|---|
| v0.7 | 训练场基础（8 算法 × 3 环境） | Released |
| v0.8 | 第 2–3 篇（MDP、Bellman 与 DP） | Released |
| v0.9 | 第 4–5 篇（MC/TD、Q-learning/SARSA） | 规划 |
| v1.0 | 第 6–8 篇（策略梯度、TRPO/PPO、DQN） | 规划 |
| v1.1 | 第 9–11 篇（AlphaGo、模型基、探索） | 规划 |
| v1.2 | 第 12–15 篇（RLHF 时代） | 规划 |
| v1.3 | 第 16–18 篇（推理与 Agent RL） | 规划 |
| v1.4 | 第 19–20 篇（总回顾时间线） | 规划 |

## 提交规范

- **Commit messages must be in English** — no Chinese in git history. Reference the column article number, e.g. `feat(dp): Value/Policy Iteration demo (column #3)`；
- 仓库的提交历史即第二条"简史时间线"；
- 教程数据在 `src/data/tutorials.ts`，每篇教程的 demo 组件按场景注册（`demo: "dp" | "mdp" | ...`）。

## 技术约定

- 路由使用 HashRouter（`base: './'`，兼容任意子路径部署：GitHub Pages / kimi 预览）；
- 不依赖任何外部 CDN 字体或资源（国内网络可达性优先）；
- JSX 属性内嵌公式时注意：JSX 属性字符串不处理反斜杠转义，公式统一走 `src/components/TeX.tsx`（KaTeX）；
- RL 算法实现保持纯 TypeScript、零 ML 依赖。

## Engineering Conventions (English-only rule)

- **No Chinese in code or git**: commit messages, code comments, identifiers, README and changelog entries are written in English. User-facing Chinese UI copy (tutorial bodies, button labels) is product language and may stay in Chinese, living in `src/data/*.ts` content fields and demo components.
- **Version discipline**: README version badge, changelog and `VERSION` file update with every release; every 0.1 bump gets a git tag (`v0.8`, `v0.9`, ...). Release notes link the matching Zhihu article.
- **Column link**: README links the companion column at <https://www.zhihu.com/column/c_2087560950404728290>.
- **Notion sync**: finalized articles go under「Notion 文章列表」in the column's Notion page.
