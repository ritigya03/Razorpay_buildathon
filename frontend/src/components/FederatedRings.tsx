import { useEffect, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../api";

const axis = { stroke: "#67728a", fontSize: 11 };
const GRID = "#e2e8f2";
const TIP = { background: "#fff", border: "1px solid #e2e8f2", borderRadius: 8, color: "#24324c" };

function PR({ label, d, strong }: { label: string; d: any; strong?: boolean }) {
  return (
    <div className="panel" style={{ background: strong ? "#eef5ff" : undefined }}>
      <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>{label}</div>
      <div className="row" style={{ gap: 18 }}>
        <div><div className="big-num">{d.precision.toFixed(2)}</div><div className="muted">precision</div></div>
        <div><div className="big-num">{d.recall.toFixed(2)}</div><div className="muted">recall</div></div>
        <div><div className="big-num">{d.f1.toFixed(2)}</div><div className="muted">F1</div></div>
      </div>
    </div>
  );
}

export function FederatedRings() {
  const [r, setR] = useState<any>(null);
  const [err, setErr] = useState(false);

  const [live, setLive] = useState<any>(null);
  const [eps, setEps] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.flRingReport().then(setR).catch(() => setErr(true));
  }, []);

  const runLive = async () => {
    setBusy(true);
    try { setLive(await api.flDetectLive(eps)); }
    catch { setLive({ error: true }); }
    finally { setBusy(false); }
  };

  if (err)
    return (
      <div className="panel">
        <h2>Federated cross-merchant ring detection — Phase 8</h2>
        <div className="muted">
          <code>report/fl_ring_metrics.json</code> not found — run <code>make fl-rings</code>{" "}
          (core <code>.venv</code>, ~1 min). Centralized vs federated + DP + poison, on the
          held-out split.
        </div>
      </div>
    );
  if (!r) return <div className="panel">loading federated ring report…</div>;

  const cen = r.centralized.combined;
  const fed = r.federated_no_dp.combined;
  const gt = r.ground_truth;
  const xmr = gt.device.cross_merchant_fraud_rings + gt.address.cross_merchant_fraud_rings;
  const smr = gt.device.single_merchant_fraud_rings + gt.address.single_merchant_fraud_rings;

  const sweep = r.dp_sweep
    .filter((x: any) => x.epsilon != null)
    .map((x: any) => ({ epsilon: x.epsilon, f1: x.combined.f1, precision: x.combined.precision, recall: x.combined.recall }))
    .sort((a: any, b: any) => a.epsilon - b.epsilon);

  const pno = r.poison_demo.hot_flood_no_defense.combined;
  const pyes = r.poison_demo.hot_flood_defended.combined;
  const poison = [
    { name: "no defense", f1: pno.f1, fill: "#d23c58" },
    { name: "robust agg", f1: pyes.f1, fill: "#0b74d1" },
    { name: "honest fed", f1: fed.f1, fill: "#4aa3c7" },
  ];

  return (
    <div className="grid" style={{ gap: 14, marginBottom: 14 }}>
      <div className="panel">
        <h2>Federated cross-merchant ring detection — the core claim, measured</h2>
        <div className="muted" style={{ fontSize: 12, marginBottom: 10 }}>
          Same LightGBM per-transaction score in both arms — so this isolates the cost of
          federating the ring layer. Entity = device / address fingerprint; a ring is
          cross-merchant when its transactions span ≥ 2 merchants. Held-out test:{" "}
          <strong>{xmr} cross-merchant fraud rings</strong> ({smr} single-merchant, for context —
          a lone merchant already catches those).
        </div>
        <div className="grid cols-2">
          <PR label="centralized — sees every transaction" d={cen} />
          <PR label="federated — 8 merchants, commit/reveal + Merkle + robust agg, no raw data shared" d={fed} strong />
        </div>
        <div className="callout" style={{ marginTop: 10, fontSize: 13 }}>
          Identical ({fed.tp} TP / {fed.fp} FP / {fed.fn} FN in both). Moving the computation
          onto the merchants costs <strong>nothing</strong>.
        </div>
      </div>

      <div className="grid cols-2">
        <div className="panel">
          <h2>Differential privacy — where the trade-off bites</h2>
          <ResponsiveContainer width="100%" height={230}>
            <LineChart data={sweep} margin={{ top: 8, right: 18, bottom: 4, left: 0 }}>
              <CartesianGrid stroke={GRID} />
              <XAxis dataKey="epsilon" type="number" scale="log" domain={["auto", "auto"]}
                     reversed tick={axis}
                     label={{ value: "ε  (lower = more private)", position: "insideBottom", offset: -2, fill: "#8a93a5", fontSize: 11 }} />
              <YAxis domain={[0, Math.max(fed.f1, 0.7) * 1.15]} tick={axis} tickFormatter={(v) => v.toFixed(2)} />
              <Tooltip contentStyle={TIP} />
              <ReferenceLine y={fed.f1} stroke="#0b74d1" strokeDasharray="4 4"
                             label={{ value: "federated, no DP", fill: "#0b74d1", fontSize: 10, position: "insideTopRight" }} />
              <Line dataKey="f1" name="F1" stroke="#02042b" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
          <div className="muted" style={{ fontSize: 12 }}>
            Free down to ε ≈ 32; below ε ≈ 16 small rings fall below the Gaussian noise floor
            (each merchant sees ~1 txn per fingerprint). An honest cost at this data scale — it
            shrinks as per-merchant volume grows.
          </div>
        </div>

        <div className="panel">
          <h2>Byzantine robustness — 1 malicious merchant</h2>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={poison} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
              <CartesianGrid stroke={GRID} />
              <XAxis dataKey="name" tick={{ ...axis, fontSize: 10 }} />
              <YAxis domain={[0, Math.max(fed.f1, 0.7) * 1.15]} tick={axis} tickFormatter={(v) => v.toFixed(2)} />
              <Tooltip contentStyle={TIP} formatter={(v: any) => Number(v).toFixed(3)} />
              <Bar dataKey="f1" isAnimationActive={false}>
                {poison.map((p, i) => <Cell key={i} fill={p.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="muted" style={{ fontSize: 12 }}>
            Merchant 0 reports every fingerprint as maximum-risk (hot-flood): precision{" "}
            {pno.precision.toFixed(2)} → {pyes.precision.toFixed(2)} once the aggregator rejects
            it as a risk-estimate outlier. Commit-mismatch is caught by the SHA-256 digest check.
          </div>
          <div className="key" style={{ marginTop: 6 }}>merkle root: {r.federated_no_dp.merkle_root}</div>
        </div>
      </div>

      {/* live protocol over the Razorpay payments */}
      <div className="panel">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h2 style={{ margin: 0 }}>Run it live — over the Razorpay test-mode payments</h2>
          <span className="row" style={{ gap: 6 }}>
            <label className="muted" style={{ fontSize: 12 }}>DP ε</label>
            {[null, 8, 4].map((e) => (
              <button key={String(e)} className={eps === e ? "primary" : ""} onClick={() => setEps(e as any)}>
                {e === null ? "off" : e}
              </button>
            ))}
            <button className="primary" onClick={runLive} disabled={busy}>
              {busy ? "…" : "run federated detection"}
            </button>
          </span>
        </div>
        <div className="muted" style={{ fontSize: 12, margin: "6px 0 10px" }}>
          Make a few coordinated payments in the Razorpay tab first (same card, different
          identities, different merchants). Each merchant node here releases only salted-HMAC
          card fingerprints + risk-bucket counts.
        </div>

        {live?.error && <div style={{ color: "var(--danger)" }}>detection failed — is the backend up?</div>}
        {live && !live.error && (
          <>
            <div className="grid cols-3" style={{ marginBottom: 10 }}>
              {live.merchant_sketches.map((sk: any) => (
                <div key={sk.merchant} className="panel" style={{ margin: 0 }}>
                  <div className="ring-head"><strong>{sk.merchant}</strong><span className="muted">{sk.payments} pmts</span></div>
                  <div className="key" style={{ margin: "4px 0" }}>commit {sk.commitment}…</div>
                  {sk.entries.map((e: any) => (
                    <div key={e.fingerprint} className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>
                      {e.fingerprint}…  [{e.buckets.join(" ")}]  n={e.count}
                    </div>
                  ))}
                </div>
              ))}
            </div>
            <div className="key" style={{ marginBottom: 8 }}>merkle root: {live.merkle_root}</div>
            <table>
              <thead><tr><th>federated result</th><th>fingerprint</th><th>payments</th><th>merchants</th><th>risk est.</th><th></th></tr></thead>
              <tbody>
                {live.federated_rings.map((rg: any) => (
                  <tr key={rg.fingerprint}>
                    <td className="muted">cross-merchant ring</td>
                    <td className="mono">{rg.fingerprint}…</td>
                    <td>{rg.payments}</td>
                    <td>{rg.merchants}</td>
                    <td className="risk">{rg.risk_estimate.toFixed(2)}</td>
                    <td>{rg.flagged ? <span className="pill won">flagged</span> : <span className="pill">watch</span>}</td>
                  </tr>
                ))}
                {live.federated_rings.length === 0 && (
                  <tr><td colSpan={6} className="muted">no cross-merchant fingerprint yet</td></tr>
                )}
              </tbody>
            </table>
            <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>
              central view (needs every raw payment pooled): {live.flagged_centralized} flagged ·
              federated: {live.flagged_federated} flagged. {live.note}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
