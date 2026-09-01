import { useEffect, useState } from "react";
import { api, type Ring, type Txn } from "../api";

const hue = (s: string) => {
  let h = 0;
  for (const c of s) h = (h * 31 + c.charCodeAt(0)) % 360;
  return h;
};

function RingGraph({ ring, members }: { ring: Ring; members: Txn[] }) {
  const W = 460, H = 340, cx = W / 2, cy = H / 2, R = 120;
  const n = members.length;
  return (
    <svg width={W} height={H} style={{ maxWidth: "100%" }}>
      {members.map((m, i) => {
        const a = (i / Math.max(n, 1)) * 2 * Math.PI - Math.PI / 2;
        const x = cx + R * Math.cos(a), y = cy + R * Math.sin(a);
        const col = m.is_fraud === 1 ? "var(--fraud)" : "#5b8def";
        return (
          <g key={m.id}>
            <line x1={cx} y1={cy} x2={x} y2={y} stroke="var(--border)" />
            <circle cx={x} cy={y} r={m.disputed ? 11 : 8} fill={col}
                    stroke={`hsl(${hue(m.card_id ?? m.id)} 70% 60%)`} strokeWidth={2.5} />
            {m.disputed && <circle cx={x} cy={y} r={15} fill="none" stroke="var(--ok)" strokeWidth={2} />}
          </g>
        );
      })}
      <circle cx={cx} cy={cy} r={26} fill="var(--panel-2)" stroke="var(--accent)" strokeWidth={2} />
      <text x={cx} y={cy - 2} textAnchor="middle" fontSize={10} fill="var(--text)">
        {ring.kind}
      </text>
      <text x={cx} y={cy + 10} textAnchor="middle" fontSize={9} fill="var(--muted)">
        {ring.distinct_cards} cards
      </text>
    </svg>
  );
}

export function Rings() {
  const [rings, setRings] = useState<Ring[]>([]);
  const [sel, setSel] = useState<number | null>(null);
  const [detail, setDetail] = useState<{ ring: Ring; members: Txn[] } | null>(null);
  const [showAll, setShowAll] = useState(false);

  useEffect(() => {
    const load = () =>
      api.rings(`?limit=60${showAll ? "" : "&flagged=true"}`).then((rs) => {
        setRings(rs);
        setSel((s) => s ?? rs[0]?.id ?? null);
      }).catch(() => {});
    load();
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, [showAll]);

  useEffect(() => {
    if (sel == null) return;
    api.ring(sel).then(setDetail).catch(() => setDetail(null));
    const t = setInterval(() => sel != null && api.ring(sel).then(setDetail).catch(() => {}), 3000);
    return () => clearInterval(t);
  }, [sel]);

  return (
    <div className="grid cols-2" style={{ alignItems: "start" }}>
      <div className="panel">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h2>Coordinated rings</h2>
          <label className="row muted" style={{ fontSize: 12 }}>
            <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} />
            include unflagged
          </label>
        </div>
        <div className="scroll">
          {rings.map((r) => (
            <div key={r.id} className="panel ring-card" style={{ margin: "8px 0", background: r.id === sel ? "var(--panel-2)" : undefined }}
                 onClick={() => setSel(r.id)}>
              <div className="ring-head">
                <strong>
                  #{r.id} · {r.kind}
                  {r.source === "razorpay" && <span className="pill" style={{ marginLeft: 6 }}>live</span>}
                  {r.n_merchants > 1 && <span className="pill" style={{ marginLeft: 4 }}>{r.n_merchants} merchants</span>}
                </strong>
                <span className={"risk " + (r.score_mean > 0.5 ? "hi" : "mid")}>{r.score_mean.toFixed(2)}</span>
              </div>
              <div className="muted" style={{ fontSize: 12 }}>
                {r.size} txns · {r.distinct_members} {r.kind === "address" ? "identities" : "accounts"} · {r.distinct_cards} cards
                {r.n_fraud > 0 && <> · <span style={{ color: "var(--fraud)" }}>{r.n_fraud} fraud</span></>}
                {r.n_disputed > 0 && <> · <span style={{ color: "var(--ok)" }}>{r.n_disputed} disputed</span></>}
              </div>
              <div className="key">{r.key}</div>
            </div>
          ))}
          {rings.length === 0 && <div className="muted">no rings yet</div>}
        </div>
      </div>

      <div className="panel">
        <h2>Ring detail</h2>
        {!detail && <div className="muted">select a ring</div>}
        {detail && (
          <>
            <RingGraph ring={detail.ring} members={detail.members} />
            <div className="muted" style={{ fontSize: 12, margin: "6px 0" }}>
              red = fraud (ground truth) · ring colour = distinct card · green halo = disputed
            </div>
            <table>
              <thead><tr><th>txn</th><th>risk</th><th>amount</th><th>card</th><th>account</th><th>truth</th></tr></thead>
              <tbody>
                {detail.members.map((m) => (
                  <tr key={m.id} className={m.is_fraud === 1 ? "fraud" : ""}>
                    <td className="mono">{m.id}</td>
                    <td className={"risk " + (m.score > 0.5 ? "hi" : "mid")}>{m.score.toFixed(2)}</td>
                    <td>{m.amount.toFixed(0)}</td>
                    <td className="mono muted">{m.card_id?.slice(0, 14)}</td>
                    <td className="mono muted">{m.uid?.slice(0, 10)}</td>
                    <td>{m.is_fraud === 1 ? <span className="pill fraud">fraud</span> : m.is_fraud === 0 ? "legit" : "?"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </div>
    </div>
  );
}
