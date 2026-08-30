import { useEffect, useState } from "react";
import { api, type Stats, type Alert } from "../api";
import { Stat } from "./Stat";

const pct = (x: number | null | undefined) => (x == null ? "—" : `${(x * 100).toFixed(1)}%`);

export function Overview({ stats }: { stats: Stats | null }) {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  useEffect(() => {
    const t = setInterval(() => api.alerts("?limit=12&kind=ring").then(setAlerts).catch(() => {}), 2000);
    return () => clearInterval(t);
  }, []);

  if (!stats) return <div className="panel">connecting to backend…</div>;
  const r = stats.replay;
  const op = stats.model_metrics.operating_point;

  return (
    <div className="grid" style={{ gap: 16 }}>
      <div className="panel">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h2>Replay feed — held-out test split</h2>
          <div className="row">
            <button onClick={() => api.replay("start")}>▶ start</button>
            <button onClick={() => api.replay("pause")}>❚❚ pause</button>
            <button onClick={() => api.replay("reset")}>↺ reset</button>
          </div>
        </div>
        <div className="row" style={{ justifyContent: "space-between", marginTop: 6 }}>
          <span className="muted">
            {r.ingested.toLocaleString()} / {r.total.toLocaleString()} transactions · virtual day{" "}
            {r.virtual_day.toFixed(1)} · {r.days_per_sec}×
          </span>
          <span className={"tag " + (r.running ? "live" : "off")}>{r.running ? "streaming" : "paused"}</span>
        </div>
        <div className="bar"><div style={{ width: `${r.progress * 100}%` }} /></div>
      </div>

      <div className="grid cols-4">
        <Stat label="transactions scored" value={stats.transactions.toLocaleString()} />
        <Stat label="flagged (advisory)" value={stats.flagged.toLocaleString()}
              sub={`${stats.flagged_true_positive.toLocaleString()} true positives`} />
        <Stat label="live precision so far" value={pct(stats.live_precision_so_far)}
              sub={op ? `held-out recall ${pct(op.recall)}` : undefined} />
        <Stat label="coordinated rings" value={stats.rings_total}
              sub={`${stats.rings_flagged} flagged`} />
      </div>

      <div className="grid cols-3">
        <Stat label="model PR-AUC (held-out)" value={stats.model_metrics.pr_auc_test?.toFixed(3) ?? "—"} />
        <Stat label="model ROC-AUC (held-out)" value={stats.model_metrics.roc_auc_test?.toFixed(3) ?? "—"} />
        <Stat label="expected-loss reduction"
              value={op ? pct(1 - op.total_expected_cost / op.do_nothing_cost) : "—"}
              sub="vs doing nothing, at the cost-optimal threshold" />
      </div>

      <div className="panel">
        <h2>Ring alerts</h2>
        {alerts.length === 0 && <div className="muted">no ring alerts yet — rings form once enough of the window has streamed</div>}
        {alerts.map((a) => (
          <div key={a.id} style={{ borderBottom: "1px solid var(--border)", padding: "6px 0", fontSize: 13 }}>
            <span className="risk hi">{a.score.toFixed(2)}</span> &nbsp;{a.summary}
          </div>
        ))}
      </div>
    </div>
  );
}
