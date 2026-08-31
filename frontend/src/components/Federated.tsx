import { useEffect, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

const axis = { stroke: "#67728a", fontSize: 11 };
const GRID = "#e2e8f2";
const TIP = { background: "#fff", border: "1px solid #e2e8f2", borderRadius: 8, color: "#24324c" };

export function Federated() {
  const [r, setR] = useState<any>(null);
  const [err, setErr] = useState(false);
  useEffect(() => {
    fetch("/api/fl-report")
      .then((x) => (x.ok ? x.json() : Promise.reject(x.status)))
      .then(setR)
      .catch(() => setErr(true));
  }, []);

  if (err)
    return (
      <div className="panel">
        <h2>Federated learning — Phase 4</h2>
        <div className="muted">
          <code>report/fl_metrics.json</code> not found. Generate it with{" "}
          <code>make fl-deps &amp;&amp; make fl</code> — a secondary experiment that runs in
          its own <code>.venv-fl</code> and does not touch the graded numbers.
        </div>
      </div>
    );
  if (!r) return <div className="panel">loading federated report…</div>;

  const cen = r.centralized.pr_auc as number;
  const fed = r.federated_no_dp.pr_auc as number;
  const sweep = [...r.dp_sweep].sort((a: any, b: any) => b.target_epsilon - a.target_epsilon);
  const target = r.dp_sweep.find((x: any) => x.target_epsilon <= 2) ?? r.dp_sweep.at(-1);
  const d = r.robustness_demo;
  const dd = d.defended;
  const yMax = Math.max(cen, fed) * 1.2;

  const poison = [
    { name: "no defense", key: "sign-flip", pr_auc: d.no_defense_plain_fedavg.pr_auc, fill: "#d23c58" },
    { name: "norm filter", key: "sign-flip ×10", pr_auc: dd.sign_flip.pr_auc, fill: "#0b74d1" },
    { name: "cosine filter", key: "flip ×1", pr_auc: dd.flip.pr_auc, fill: "#0b74d1" },
    { name: "commit/reveal", key: "last-mover", pr_auc: dd.last_mover.pr_auc, fill: "#0b74d1" },
  ];
  const allRej = [
    ...dd.sign_flip.rejections, ...dd.flip.rejections, ...dd.last_mover.rejections,
  ];

  return (
    <div className="grid" style={{ gap: 14 }}>
      <div className="panel">
        <h2>Privacy-preserving adapter — federated vs. centralized</h2>
        <div className="row" style={{ gap: 24 }}>
          <div><div className="big-num">{cen.toFixed(3)}</div><div className="muted">centralized PR-AUC</div></div>
          <div><div className="big-num">{fed.toFixed(3)}</div><div className="muted">federated, no DP</div></div>
          {target && (
            <div><div className="big-num">{target.pr_auc.toFixed(3)}</div>
              <div className="muted">federated @ ε≈{target.target_epsilon} (spent ≤{target.spent_epsilon_max})</div></div>
          )}
          <div><div className="big-num">{r.meta.n_merchants}</div>
            <div className="muted">merchants · {r.meta.fl_rounds} rounds</div></div>
        </div>
        <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>{r.meta.disclaimer}</div>
      </div>

      <div className="grid cols-2">
        <div className="panel">
          <h2>Utility vs. privacy budget ε</h2>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={sweep} margin={{ top: 8, right: 18, bottom: 4, left: 0 }}>
              <CartesianGrid stroke={GRID} />
              <XAxis dataKey="target_epsilon" type="number" scale="log" domain={["auto", "auto"]}
                     reversed tick={axis}
                     label={{ value: "ε  (lower = more private)", position: "insideBottom", offset: -2, fill: "#8a93a5", fontSize: 11 }} />
              <YAxis domain={[0, yMax]} tick={axis} tickFormatter={(v) => v.toFixed(2)} />
              <Tooltip contentStyle={TIP} />
              <ReferenceLine y={cen} stroke="#0b74d1" strokeDasharray="4 4"
                             label={{ value: "centralized", fill: "#0b74d1", fontSize: 10, position: "insideTopRight" }} />
              <ReferenceLine y={fed} stroke="#4aa3c7" strokeDasharray="4 4"
                             label={{ value: "federated, no DP", fill: "#4aa3c7", fontSize: 10, position: "insideBottomRight" }} />
              <Line dataKey="pr_auc" stroke="#02042b" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
          <table style={{ marginTop: 8, fontSize: 12 }}>
            <thead><tr><th>target ε</th><th>spent ε</th><th>noise mult.</th><th>PR-AUC</th><th>Δ vs central</th></tr></thead>
            <tbody>
              {r.dp_sweep.map((x: any) => (
                <tr key={x.target_epsilon}>
                  <td>{x.target_epsilon}</td>
                  <td className="mono">{x.spent_epsilon_max}</td>
                  <td className="mono muted">{x.noise_multiplier_range?.join("–")}</td>
                  <td className="risk">{x.pr_auc.toFixed(3)}</td>
                  <td className="mono">{x.delta_pr_auc_vs_centralized > 0 ? "+" : ""}{x.delta_pr_auc_vs_centralized}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="panel">
          <h2>Byzantine robustness — {d.malicious_merchants.length}/{r.meta.n_merchants} merchants malicious</h2>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={poison} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
              <CartesianGrid stroke={GRID} />
              <XAxis dataKey="name" tick={{ ...axis, fontSize: 10 }} />
              <YAxis domain={[0, yMax]} tick={axis} tickFormatter={(v) => v.toFixed(2)} />
              <Tooltip contentStyle={TIP} formatter={(v: any, _n, p: any) => [Number(v).toFixed(3), p.payload.key]} />
              <ReferenceLine y={fed} stroke="#4aa3c7" strokeDasharray="4 4"
                             label={{ value: "honest federated", fill: "#4aa3c7", fontSize: 10, position: "insideTopRight" }} />
              <Bar dataKey="pr_auc" isAnimationActive={false}>
                {poison.map((p, i) => <Cell key={i} fill={p.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
            three attacks, one defended aggregator: <strong>sign-flip ×10</strong> → norm test,
            {" "}<strong>flip ×1</strong> → cosine test, <strong>last-mover</strong> (honest
            commit, poisoned reveal) → SHA-256 mismatch. {allRej.length} updates rejected in total.
          </div>
          <div className="key" style={{ marginTop: 6 }}>
            merkle root (round 1): {r.federated_no_dp.merkle_root_round1}
          </div>
          {allRej.length > 0 && (
            <table style={{ marginTop: 8, fontSize: 12 }}>
              <thead><tr><th>round</th><th>merchant</th><th>reason</th><th>detail</th></tr></thead>
              <tbody>
                {allRej.slice(0, 8).map((x: any, i: number) => (
                  <tr key={i}>
                    <td>{x.round}</td><td>{x.merchant}</td><td>{x.reason}</td>
                    <td className="mono muted">{x.norm_ratio ?? x.cos_distance ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      <div className="panel">
        <h2>Merchant partition — non-IID (hash of card1)</h2>
        <table style={{ maxWidth: 560 }}>
          <thead><tr><th>merchant</th><th>rows</th><th>fraud rows</th><th>fraud rate</th></tr></thead>
          <tbody>
            {r.federated_no_dp.per_merchant.map((m: any) => (
              <tr key={m.merchant}>
                <td>{m.merchant}</td>
                <td>{m.rows.toLocaleString()}</td>
                <td>{m.fraud_rows.toLocaleString()}</td>
                <td className="risk">{(m.fraud_rate * 100).toFixed(2)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
          {r.meta.framework}. DP: {r.meta.dp.mechanism}, δ={r.meta.dp.delta}, clip={r.meta.dp.max_grad_norm}.
          Model selected on validation PR-AUC (round {r.federated_no_dp.best_round}).
        </div>
      </div>
    </div>
  );
}
