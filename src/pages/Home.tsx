import { Link } from "react-router";
import { ArrowRight, ExternalLink } from "lucide-react";
import { algorithms } from "@/data/algorithms";
import { tutorials } from "@/data/tutorials";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

export default function Home() {
  return (
    <div className="space-y-10">
      {/* Hero：专栏联动 Banner */}
      <section className="relative overflow-hidden rounded-2xl border border-border">
        <img src="/column-cover.png" alt="强化学习简史专栏封面" className="w-full object-cover" />
        <div className="absolute inset-0 bg-gradient-to-r from-black/85 via-black/50 to-transparent flex flex-col justify-center px-6 md:px-10">
          <Badge className="w-fit mb-3 bg-accent-yellow text-black hover:bg-accent-yellow">专栏配套沙盒</Badge>
          <h1 className="text-2xl md:text-4xl font-bold text-white leading-tight">
            从 Bellman 方程到 Agentic RL
          </h1>
          <p className="mt-2 text-sm md:text-base text-white/80 max-w-xl">
            「强化学习简史」的交互式演武场——专栏读到的每一个关键算法，
            都能在这里亲手拨动、逐帧观察。
          </p>
          <div className="mt-4 flex flex-wrap gap-2">
            <Link to="/playground">
              <Button size="sm" className="bg-accent-purple hover:bg-accent-purple/90">
                进入训练场 <ArrowRight className="w-4 h-4 ml-1" />
              </Button>
            </Link>
            <Link to="/tutorials">
              <Button size="sm" variant="outline" className="border-white/30 text-white hover:bg-white/10 hover:text-white">
                交互式教程
              </Button>
            </Link>
            <a href="https://github.com/MagicLeonStudio/MagicRLSandBox" target="_blank" rel="noreferrer">
              <Button size="sm" variant="ghost" className="text-white/80 hover:text-white">
                GitHub <ExternalLink className="w-3.5 h-3.5 ml-1" />
              </Button>
            </a>
          </div>
        </div>
      </section>

      {/* 教程入口 */}
      <section className="space-y-4">
        <div className="flex items-end justify-between">
          <h2 className="text-xl font-bold">专栏配套教程</h2>
          <Link to="/tutorials" className="text-sm text-accent-purple hover:underline">全部 →</Link>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {tutorials.map(t => (
            <Link key={t.id} to={`/tutorials/${t.id}`}>
              <Card className="h-full transition-colors hover:border-accent-purple/60 hover:shadow-glow-purple">
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary">第 {t.no} 篇</Badge>
                    <span className="text-xs text-muted-foreground">{t.act}</span>
                  </div>
                  <CardTitle className="text-lg leading-snug mt-2">{t.title}</CardTitle>
                  <CardDescription>{t.subtitle}</CardDescription>
                </CardHeader>
              </Card>
            </Link>
          ))}
        </div>
      </section>

      {/* 算法矩阵 */}
      <section className="space-y-4">
        <div className="flex items-end justify-between">
          <div>
            <h2 className="text-xl font-bold">算法矩阵</h2>
            <p className="text-sm text-muted-foreground mt-1">纯 TypeScript 手写实现，零 ML 依赖 · {algorithms.length} 个算法 × 3 个环境</p>
          </div>
          <Link to="/playground" className="text-sm text-accent-purple hover:underline">进入训练场 →</Link>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {algorithms.map(a => (
            <Link key={a.id} to="/playground">
              <Card className="h-full transition-colors hover:border-accent-yellow/50 hover:shadow-glow-yellow">
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-base">{a.name}</CardTitle>
                    <span className="text-xs font-mono text-accent-yellow">{a.year}</span>
                  </div>
                  <Badge variant="outline" className="w-fit text-[10px]">{a.category}</Badge>
                </CardHeader>
                <CardContent className="pt-0">
                  <p className="text-xs text-muted-foreground line-clamp-2">{a.description}</p>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      </section>

      <footer className="border-t border-border/60 pt-6 pb-2 flex flex-wrap items-center justify-between gap-3 text-sm text-muted-foreground">
        <span>MagicLeon · Magic Leon Studio（碼良計畫）</span>
        <div className="flex gap-4">
          <a className="hover:text-accent-purple" href="https://www.zhihu.com/people/MagicLeon" target="_blank" rel="noreferrer">知乎专栏</a>
          <a className="hover:text-accent-purple" href="https://github.com/MagicLeonStudio/MagicRLSandBox" target="_blank" rel="noreferrer">GitHub</a>
        </div>
      </footer>
    </div>
  );
}
