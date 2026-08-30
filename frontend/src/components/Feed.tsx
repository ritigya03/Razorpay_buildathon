import { useEffect, useState } from "react";
import { api, type Txn } from "../api";

const money = (n: number) => n.toLocaleString(undefined, { maximumFractionDigits: 0 });

export function Feed() {
  const [txns, setTxns] = useState<Txn[]>([]);
  const [onlyFlagged, setOnlyFlagged] = useState(false);

  useEffect(() => {
    const load = () =>
      api.transactions(`?limit=80${onlyFlagged ? "&flagged=true" : ""}`).then(setTxns).catch(() => {});
    load();
    const t = setInterval(load, 2000);
    return () => clearInterval(t);
  }, [onlyFlagged]);

  return (
    <div className="panel">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h2>Live transaction feed</h2>
        <label className="row muted" style={{ fontSize: 12 }}>
          <input type="checkbox" checked={onlyFlagged} onChange={(e) => setOnlyFlagged(e.target.checked)} />
          flagged only
        </label>
      </div>
      <div className="scroll">
        <table>
          <thead>
            <tr>
              <th>id</th><th>src</th><th>risk</th><th>amount</th><th>email</th>
              <th>device</th><th>ring</th><th>truth</th>
            </tr>
          </thead>
          <tbody>
            {txns.map((t) => (
              <tr key={t.id} className={(t.is_fraud === 1 ? "fraud " : "") + (t.flagged ? "flagged" : "")}>
                <td className="mono">{t.id}</td>
                <td><span className={"pill " + t.scorer}>{t.scorer}</span></td>
                <td className={"risk " + (t.score > 0.5 ? "hi" : t.score > 0.15 ? "mid" : "")}>{t.score.toFixed(2)}</td>
                <td>{money(t.amount)}</td>
                <td className="muted">{t.email_domain ?? "—"}</td>
                <td className="muted mono" title={t.device_id ?? ""}>
                  {t.device_id ? t.device_id.slice(0, 22) : "—"}
                </td>
                <td>{t.ring_id ? <span className="pill">#{t.ring_id}</span> : "—"}</td>
                <td>
                  {t.is_fraud === 1 ? <span className="pill fraud">fraud</span>
                    : t.is_fraud === 0 ? <span className="pill">legit</span> : <span className="muted">?</span>}
                  {t.disputed && <span className="pill lost" style={{ marginLeft: 4 }}>disputed</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
