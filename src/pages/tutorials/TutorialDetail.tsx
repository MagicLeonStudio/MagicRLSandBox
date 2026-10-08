import { useParams, Link } from "react-router";
import { ArrowLeft } from "lucide-react";
import { tutorials } from "@/data/tutorials";
import TeX from "@/components/TeX";
import DPDemo from "@/components/DPDemo";
import MCTDDemo from "@/components/MCTDDemo";
import CliffWalkDemo from "@/components/CliffWalkDemo";
import ValueHeatmap from "@/components/ValueHeatmap";
import { Button } from "@/components/ui/button";

const ZERO25 = new Array(25).fill(0);

export default function TutorialDetail() {
  const { id } = useParams();
  const t = tutorials.find(x => x.id === id);

  if (!t) {
    return (
      <div className="space-y-4">
        <p className="text-muted-foreground">教程不存在。</p>
        <Link to="/tutorials"><Button variant="outline" size="sm"><ArrowLeft className="w-4 h-4 mr-1" />返回教程列表</Button></Link>
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto space-y-8">
      <div className="space-y-2">
        <Link to="/tutorials" className="text-sm text-muted-foreground hover:text-accent-purple inline-flex items-center gap-1">
          <ArrowLeft className="w-3.5 h-3.5" />全部教程
        </Link>
        <div className="flex items-center gap-3 text-sm text-muted-foreground">
          <span>{t.act}</span>
          <span>·</span>
          <span>专栏第 {t.no} 篇</span>
        </div>
        <h1 className="text-3xl font-bold leading-tight">{t.title}</h1>
        <p className="text-muted-foreground">{t.subtitle}</p>
        {t.zhihuUrl ? (
          <a href={t.zhihuUrl} target="_blank" rel="noreferrer" className="text-sm text-accent-yellow hover:underline">
            在知乎阅读完整专栏文章 →
          </a>
        ) : (
          <span className="text-sm text-muted-foreground">专栏正文撰写中</span>
        )}
      </div>

      {t.sections.map((sec, i) => (
        <section key={i} className="space-y-3">
          <h2 className="text-xl font-semibold border-l-2 border-accent-yellow pl-3">{sec.heading}</h2>
          <p className="text-foreground/90 leading-relaxed text-[15px]">{sec.body}</p>
          {sec.formula && <TeX math={sec.formula} block />}
          {sec.demo === "dp" && <DPDemo />}
          {sec.demo === "mctd" && <MCTDDemo />}
          {sec.demo === "cliff" && <CliffWalkDemo />}
          {sec.demo === "mdp" && (
            <div className="flex justify-center bg-black/30 rounded-lg p-4">
              <ValueHeatmap size={5} values={ZERO25} goal={24} showValues={false} />
            </div>
          )}
        </section>
      ))}
    </div>
  );
}
