import { useEffect, useState } from "react";
import {
  CartesianGrid, Line, LineChart, ReferenceDot, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../api";

const axis = { stroke: "#67728a", fontSize: 11 };
const GRID = "#e2e8f2";
const TIP = { background: "#fff", border: "1px solid #e2e8f2", borderRadius: 8, color: "#24324c" };

export function Metrics() {
  const [rep, setRep] = useState<any>(null);
  useEffect(() => { api.report().then(setRep).catch(() => {}); }, []);
  if (!rep?.metrics) return <div className="panel">loading report…</div>;

  const m = rep.metrics;
  const curve = [...m.cost_curve_test].sort((a: any, b: any) => a.recall - b.recall);
  const op = m.operating_point.test;
  const dev = rep.ring_metrics?.rings?.device;
  const addr = rep.ring_metrics?.rings?.address;

  return (
    <div className="grid" style={{ gap: 14 }}>
      <div className="panel">
        <h2>Transaction model — held-out test</h2>
        <div className="row" style={{ gap: 24 }}>
          <div><div className="big-num">{m.pr_auc.test.toFixed(3)}</div><div className="muted">PR-AUC</div></div>
          <div><div className="big-num">{m.roc_auc.test.toFixed(3)}</div><div className="muted">ROC-AUC</div></div>
          <div><div className="big-num">{(op.recall * 100).toFixed(0)}%</div><div className="muted">recall @ cost-optimal</div></div>
          <div><div className="big-num">{(op.precision * 100).toFixed(0)}%</div><div className="muted">precision @ cost-optimal</div></div>
          <div><div className="big-num">{((1 - op.total_expected_cost / op.do_nothing_cost) * 100).toFixed(0)}%</div>
            <div className="muted">expected-loss reduction</div></div>
        </div>
      </div>

      <div className="grid cols-2">
        <div className="panel">
          <h2>Precision — recall (test)</h2>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={curve} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
              <CartesianGrid stroke={GRID} />
              <XAxis dataKey="recall" type="number" domain={[0, 1]} tick={axis}
                     tickFormatter={(v) => v.toFixed(1)} label={{ value: "recall", position: "insideBottom", offset: -2, fill: "#8a93a5", fontSize: 11 }} />
              <YAxis dataKey="precision" domain={[0, 1]} tick={axis} tickFormatter={(v) => v.toFixed(1)} />
              <Tooltip contentStyle={TIP} />
              <Line dataKey="precision" stroke="#0b74d1" dot={false} strokeWidth={2} isAnimationActive={false} />
              <ReferenceDot x={op.recall} y={op.precision} r={5} fill="#b0710a" stroke="none" />
            </LineChart>
          </ResponsiveContainer>
          <div className="muted" style={{ fontSize: 12 }}>amber dot = cost-optimal operating point</div>
        </div>

        <div className="panel">
          <h2>Expected loss vs recall (test)</h2>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={curve} margin={{ top: 8, right: 12, bottom: 4, left: 8 }}>
              <CartesianGrid stroke={GRID} />
              <XAxis dataKey="recall" type="number" domain={[0, 1]} tick={axis} tickFormatter={(v) => v.toFixed(1)} />
              <YAxis tick={axis} tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`} />
              <Tooltip contentStyle={TIP}
                       formatter={(v: any) => Number(v).toLocaleString()} />
              <Line dataKey="cost" stroke="#d23c58" dot={false} strokeWidth={2} isAnimationActive={false} />
              <ReferenceDot x={op.recall} y={op.total_expected_cost} r={5} fill="#b0710a" stroke="none" />
            </LineChart>
          </ResponsiveContainer>
          <div className="muted" style={{ fontSize: 12 }}>
            cost = review_cost·FP + Σ(missed-fraud amount). "do nothing" = {op.do_nothing_cost.toLocaleString()}
          </div>
        </div>
      </div>

      {dev && (
        <div className="panel">
          <h2>Ring engine — held-out test</h2>
          <table style={{ maxWidth: 560 }}>
            <thead><tr><th>ring type</th><th>groups</th><th>fraud rings</th><th>flagged</th>
              <th>precision</th><th>recall</th><th>F1</th></tr></thead>
            <tbody>
              {[["device", dev], ["address", addr]].filter(([, d]) => d).map(([k, d]: any) => (
                <tr key={k}>
                  <td>{k}</td><td>{d.groups}</td><td>{d.true_fraud_rings}</td><td>{d.rings_flagged}</td>
                  <td className="risk">{d.ring_precision.toFixed(2)}</td>
                  <td className="risk">{d.ring_recall.toFixed(2)}</td>
                  <td>{d.ring_f1.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
            explicit ring <em>features</em> gave no PR-AUC lift (+0.003) — the ring engine's value is triage:
            {" "}{rep.metrics.operating_point.test.fp + rep.metrics.operating_point.test.tp} txn alerts →{" "}
            {(dev.rings_flagged + (addr?.rings_flagged ?? 0))} ring alerts.
          </div>
        </div>
      )}
    </div>
  );
}
