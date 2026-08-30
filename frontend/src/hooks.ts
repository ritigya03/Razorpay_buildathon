import { useEffect, useRef, useState } from "react";

/** Count from 0 to `target` once, on mount (or when `run` flips true). */
export function useCountUp(target: number, opts: { duration?: number; decimals?: number; run?: boolean } = {}) {
  const { duration = 1200, decimals = 0, run = true } = opts;
  const [val, setVal] = useState(0);
  useEffect(() => {
    if (!run) return;
    let raf = 0;
    const t0 = performance.now();
    const tick = (t: number) => {
      const p = Math.min((t - t0) / duration, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      setVal(target * eased);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, duration, run]);
  return val.toFixed(decimals);
}

/** Adds the `in` class when the element scrolls into view (pairs with .reveal). */
export function useReveal<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([e]) => e.isIntersecting && (el.classList.add("in"), io.disconnect()),
      { threshold: 0.15 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return ref;
}
