import { useEffect, useRef } from "react";

/** Lightweight animated node/edge network — evokes a fraud ring graph. */
export function NetworkCanvas() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current!;
    const ctx = cv.getContext("2d")!;
    let raf = 0;
    let w = 0, h = 0;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    type P = { x: number; y: number; vx: number; vy: number };
    let pts: P[] = [];

    const resize = () => {
      w = cv.clientWidth; h = cv.clientHeight;
      cv.width = w * DPR; cv.height = h * DPR;
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      const n = Math.round(Math.min(90, (w * h) / 18000));
      pts = Array.from({ length: n }, () => ({
        x: Math.random() * w, y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.35, vy: (Math.random() - 0.5) * 0.35,
      }));
    };

    const draw = () => {
      ctx.clearRect(0, 0, w, h);
      for (const p of pts) {
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0 || p.x > w) p.vx *= -1;
        if (p.y < 0 || p.y > h) p.vy *= -1;
      }
      for (let i = 0; i < pts.length; i++) {
        for (let j = i + 1; j < pts.length; j++) {
          const a = pts[i], b = pts[j];
          const d = Math.hypot(a.x - b.x, a.y - b.y);
          if (d < 130) {
            ctx.strokeStyle = `rgba(51,149,255,${0.16 * (1 - d / 130)})`;
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
          }
        }
      }
      for (const p of pts) {
        ctx.fillStyle = "rgba(120,200,255,0.75)";
        ctx.beginPath(); ctx.arc(p.x, p.y, 1.7, 0, 7); ctx.fill();
      }
      raf = requestAnimationFrame(draw);
    };

    resize();
    window.addEventListener("resize", resize);
    if (reduce) draw(); // one frame
    else raf = requestAnimationFrame(draw);
    return () => { cancelAnimationFrame(raf); window.removeEventListener("resize", resize); };
  }, []);

  return <canvas ref={ref} className="net" />;
}
