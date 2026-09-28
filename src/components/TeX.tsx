import { useMemo } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";

/** 知乎文公式与沙盒内公式统一走 KaTeX 渲染，语法与 peco-equation 的 LaTeX 子集一致 */
export default function TeX({ math, block = false }: { math: string; block?: boolean }) {
  const html = useMemo(
    () =>
      katex.renderToString(math, {
        throwOnError: false,
        displayMode: block,
      }),
    [math, block]
  );
  return block ? (
    <div className="my-3 overflow-x-auto text-center" dangerouslySetInnerHTML={{ __html: html }} />
  ) : (
    <span dangerouslySetInnerHTML={{ __html: html }} />
  );
}
