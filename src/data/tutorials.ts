/** 教程数据 —— 与知乎专栏「强化学习简史」逐篇对应 */
export interface TutorialSection {
  heading: string;
  body: string;
  formula?: string;       // KaTeX 渲染
  demo?: "dp" | "mdp";    // 嵌入的交互演示组件
}

export interface Tutorial {
  id: string;
  no: number;             // 专栏篇号
  title: string;
  subtitle: string;
  act: string;            // 所属幕
  zhihuUrl?: string;
  ready: boolean;         // 专栏正文是否已发布
  sections: TutorialSection[];
}

export const tutorials: Tutorial[] = [
  {
    id: "mdp",
    no: 2,
    title: "马尔可夫决策过程：RL 世界的元素周期表",
    subtitle: "状态、动作、奖励、折扣与策略——五元组描述一切序贯决策",
    act: "第一幕 · 地基",
    ready: false,
    sections: [
      {
        heading: "五元组 ⟨S, A, P, R, γ⟩",
        body: "马尔可夫决策过程（MDP）是强化学习的问题语言。智能体在状态 s ∈ S 下依据策略 π 选择动作 a ∈ A，环境按转移核 P(s′|s,a) 给出下一状态与奖励 r ∼ R(s,a)。折扣因子 γ ∈ [0,1) 决定未来奖励的现值。马尔可夫性意味着下一状态只依赖当前状态与动作，与历史无关。",
        formula: "M=\\langle\\mathcal{S},\\mathcal{A},P,R,\\gamma\\rangle",
      },
      {
        heading: "回报与目标",
        body: "智能体追求的不是即时奖励，而是折扣回报 G_t——从 t 时刻起所有未来奖励的折扣和。强化学习的目标形式化为寻找使期望回报最大化的策略。",
        formula: "G_t=\\sum_{k=0}^{\\infty}\\gamma^k R_{t+k+1},\\qquad \\pi^*=\\arg\\max_{\\pi}\\ \\mathbb{E}_{\\pi}[G_t]",
      },
      {
        heading: "GridWorld：最小的 MDP 试验场",
        body: "下面是一个 5×5 GridWorld：左上角为起点 S，右下角 ★ 为目标（奖励 +10），中间散落着红色障碍（撞上 −5 并原地不动），每走一步付出 −0.1 的时间惩罚，撞墙同样原地不动。它小到你口算就能验证，却包含 MDP 的全部要素——状态、动作、转移、奖励、折扣。亲手走几局试试：最优策略会贴着墙走还是对角走？每步 −0.1 的惩罚在塑造策略形状上起了什么作用？本沙盒中的动态规划与 TD 算法都会在这座小世界里演示。",
        demo: "mdp",
      },
    ],
  },
  {
    id: "bellman-dp",
    no: 3,
    title: "Bellman 方程与动态规划：一切的源头，一切的诅咒",
    subtitle: "最优性原理、价值迭代与策略迭代——以及维度灾难",
    act: "第一幕 · 地基",
    ready: false,
    sections: [
      {
        heading: "Bellman 最优方程",
        body: "Bellman 最优性方程指出：最优价值函数是其自身经 max-算子变换的不动点。这个自指结构是 70 年来几乎所有 RL 算法的代数原点——价值迭代直接把它变成迭代格式，Q-learning 采样近似它，深度 RL 用神经网络参数化它。",
        formula: "V^*(s)=\\max_a\\sum_{s\'}P(s\'|s,a)\\Big[R(s,a,s\')+\\gamma V^*(s\')\\Big]",
      },
      {
        heading: "价值迭代（VI）",
        body: "反复应用 Bellman 最优算子 T：每轮对所有状态做同步扫描更新。由于 T 是 γ-压缩映射（Banach 不动点定理保证），迭代必然收敛到唯一不动点 V*。残差 Δ_k = max_s|V_{k+1}(s)−V_k(s)| 以 γ 的速率指数衰减——拖动下方 γ 滑杆，观察收敛速度如何随折扣因子变化。",
        formula: "V_{k+1}(s)\\leftarrow\\max_a\\sum_{s\'}P(s\'|s,a)\\big[R(s,a,s\')+\\gamma V_k(s\')\\big]",
        demo: "dp",
      },
      {
        heading: "策略迭代（PI）",
        body: "策略迭代交替执行两步：策略评估（把当前策略的价值 V^π 解出来）与策略改进（对 V^π 贪心化）。有限 MDP 上它有限步内必然收敛到最优策略。切换上方 Tab 对比两种算法在同一张地图上的收敛过程——VI 是'每轮小步靠近'，PI 是'评估到底再大幅改进'。",
        formula: "V^{\\pi}(s)=\\sum_{s\'}P\\big(s\'|s,\\pi(s)\\big)\\big[R+\\gamma V^{\\pi}(s\')\\big],\\qquad \\pi'(s)=\\arg\\max_a Q^{\\pi}(s,a)",
      },
      {
        heading: "维度灾难：历史的反击",
        body: "DP 要求已知模型 P、R 且状态有限。状态数随变量数指数爆炸（Bellman 称之为 curse of dimensionality），这正是后文采样方法（MC/TD）、函数逼近与深度学习登场的历史动力。",
      },
    ],
  },
];
