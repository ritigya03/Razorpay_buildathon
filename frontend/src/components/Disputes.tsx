import { useEffect, useState } from "react";
import { api, type Dispute, type Stats } from "../api";

export function Disputes({ stats }: { stats: Stats | null }) {
  const [disputes, setDisputes] = useState<Dispute[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const load = () => api.disputes().then(setDisputes).catch(() => {});
  useEffect(() => {
    load();
    const t = setInterval(load, 2500);
    return () => clearInterval(t);
  }, []);

  const simulate = async () => {
    setBusy(true); setErr(null);
    try { await api.simulateDispute(); await load(); }
    catch (e) { setErr(String(e)); }
    finally { setBusy(false); }
  };

  const flagged = disputes.filter((d) => d.was_flagged);
  const avgLead = flagged.length
    ? flagged.reduce((s, d) => s + (d.lead_time_hours ?? 0), 0) / flagged.length : 0;

  return (
    <div className="grid" style={{ gap: 14 }}>
      <div className="panel row" style={{ justifyContent: "space-between" }}>
        <div>
          <h2>Dispute / chargeback loop</h2>
          <span className="muted" style={{ fontSize: 13 }}>
            Razorpay disputes are the ground-truth signal. A simulated dispute targets a
            flagged fraudulent transaction sitting in a flagged ring.
          </span>
        </div>
        <button className="primary" onClick={simulate} disabled={busy}>
          {busy ? "…" : "simulate a dispute"}
        </button>
      </div>
      {err && <div className="panel" style={{ color: "var(--danger)" }}>{err}</div>}

      {flagged.length > 0 && (
        <div className="panel callout">
          <div className="big-num">{avgLead.toFixed(0)} h</div>
          <div className="muted">
            average lead time — Sentinel flagged these transactions this long before the dispute was raised.
            {stats && ` ${stats.disputes_flagged_in_advance}/${stats.disputes} disputes were flagged in advance.`}
          </div>
        </div>
      )}

      <div className="panel">
        <table>
          <thead>
            <tr><th>dispute</th><th>payment</th><th>src</th><th>phase</th><th>amount</th>
              <th>flagged first?</th><th>lead time</th><th>ring</th><th>status</th></tr>
          </thead>
          <tbody>
            {disputes.map((d) => (
              <tr key={d.id}>
                <td className="mono">{d.id.slice(0, 16)}</td>
                <td className="mono">{d.payment_id}</td>
                <td className="muted">{d.source}</td>
                <td className="muted">{d.phase ?? "—"}</td>
                <td>{d.amount.toFixed(0)}</td>
                <td>{d.was_flagged
                  ? <span className="pill won">yes</span>
                  : <span className="pill lost">missed</span>}</td>
                <td>{d.lead_time_hours != null ? `${d.lead_time_hours.toFixed(1)} h` : "—"}</td>
                <td>{d.ring_id ? <span className="pill">#{d.ring_id}</span> : "—"}</td>
                <td className="muted">{d.status}</td>
              </tr>
            ))}
            {disputes.length === 0 && <tr><td colSpan={9} className="muted">no disputes yet</td></tr>}
          </tbody>
        </table>
      </div>
    </div>
  );
}
