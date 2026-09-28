import { Link } from "react-router";
import { tutorials } from "@/data/tutorials";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export default function TutorialList() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">交互式教程</h1>
        <p className="text-muted-foreground text-sm mt-1">
          「强化学习简史」专栏配套 —— 每一篇都有可以亲手拨动的算法演示。
        </p>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {tutorials.map(t => (
          <Link key={t.id} to={`/tutorials/${t.id}`}>
            <Card className="h-full transition-colors hover:border-accent-purple/60 hover:shadow-glow-purple">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <Badge variant="secondary">第 {t.no} 篇</Badge>
                  <span className="text-xs text-muted-foreground">{t.act}</span>
                </div>
                <CardTitle className="text-lg leading-snug mt-2">{t.title}</CardTitle>
                <CardDescription>{t.subtitle}</CardDescription>
              </CardHeader>
              <CardContent>
                <span className={`text-xs font-mono ${t.ready ? "text-accent-yellow" : "text-muted-foreground"}`}>
                  {t.ready ? "● 专栏正文已发布" : "○ 正文撰写中 · 演示抢先体验"}
                </span>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}
