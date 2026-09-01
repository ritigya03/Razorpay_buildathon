import { useEffect, useState } from "react";
import { api, type Ring, type Txn } from "../api";

declare global {
  interface Window { Razorpay?: any; }
}

const MERCHANTS = ["Merchant A", "Merchant B", "Merchant C"];
const IDENTITIES = [
  { label: "Aarav", email: "aarav.demo@gmail.com", contact: "9000000001" },
  { label: "Isha", email: "isha.demo@outlook.com", contact: "9000000002" },
  { label: "Kabir", email: "kabir.demo@protonmail.com", contact: "9000000003" },
];

export function RazorpayPanel({ keyId }: { keyId: string | null }) {
  const [amount, setAmount] = useState(500);
  const [merchant, setMerchant] = useState(MERCHANTS[0]);
  const [ident, setIdent] = useState(IDENTITIES[0]);
  const [log, setLog] = useState<string[]>([]);
  const [live, setLive] = useState<Txn[]>([]);
  const [rings, setRings] = useState<Ring[]>([]);
  const add = (s: string) => setLog((l) => [new Date().toLocaleTimeString() + "  " + s, ...l].slice(0, 12));

  useEffect(() => {
    const load = () => {
      api.transactions("?source=razorpay&limit=20").then(setLive).catch(() => {});
      api.rings("?source=razorpay&limit=20").then(setRings).catch(() => {});
    };
    load();
    const t = setInterval(load, 2500);
    return () => clearInterval(t);
  }, []);

  const pay = async () => {
    try {
      const order = await api.createOrder(amount * 100, { sentinel_merchant: merchant });
      add(`order ${order.id} · ${merchant} · ${ident.label} (₹${amount})`);
      if (!window.Razorpay) { add("checkout.js not loaded"); return; }
      const rzp = new window.Razorpay({
        key: keyId, order_id: order.id, amount: order.amount, currency: "INR",
        name: merchant, description: "test payment",
        prefill: { email: ident.email, contact: ident.contact },
        handler: async (resp: any) => {
          add(`paid — ${resp.razorpay_payment_id}, ingesting…`);
          try {
            // ingest immediately (no webhook / ngrok needed for local testing).
            // card metadata = the standard Razorpay test Visa card.
            const r = await api.verify({
              razorpay_order_id: order.id,
              razorpay_payment_id: resp.razorpay_payment_id,
              razorpay_signature: resp.razorpay_signature,
              payment: {
                id: resp.razorpay_payment_id, amount: amount * 100, method: "card",
                email: ident.email, contact: ident.contact,
                card: { last4: "1111", network: "Visa", type: "credit", international: false },
                notes: { sentinel_merchant: merchant },
              },
            });
            add(`ingested · risk ${Number(r.score ?? 0).toFixed(2)}`);
          } catch (e) {
            add("verify/ingest failed: " + String(e));
          }
        },
        modal: { ondismiss: () => add("checkout dismissed") },
        theme: { color: "#0d94fb" },
      });
      rzp.open();
    } catch (e) {
      add("error: " + String(e));
    }
  };

  const seed = async (kind: "shared_card" | "carding") => {
    try {
      const r = await api.demoScenario(kind, 4);
      add(`seeded ${kind}: ${r.payments.length} payments, ${r.flagged_rings.length} ring(s) flagged`);
    } catch (e) {
      add("seed error: " + String(e));
    }
  };

  return (
    <div className="grid cols-2" style={{ alignItems: "start" }}>
      <div className="panel">
        <h2>Live test-mode payment</h2>
        {!keyId && <div style={{ color: "var(--warn)" }}>Razorpay keys not configured on the backend.</div>}

        <div className="muted" style={{ fontSize: 12, margin: "6px 0 10px" }}>
          Pay as a coordinated actor would: <strong>same test card, switch the customer
          identity each time</strong> → a shared-card ring forms across merchants. Same
          identity, switch cards → a carding ring. Detection is by the live rules
          scorer, not the graded model.
        </div>

        <div className="row" style={{ gap: 6, marginBottom: 8 }}>
          <span className="muted" style={{ fontSize: 12, width: 62 }}>merchant</span>
          {MERCHANTS.map((m) => (
            <button key={m} className={m === merchant ? "primary" : ""} onClick={() => setMerchant(m)}>
              {m.replace("Merchant ", "")}
            </button>
          ))}
        </div>
        <div className="row" style={{ gap: 6, marginBottom: 8 }}>
          <span className="muted" style={{ fontSize: 12, width: 62 }}>identity</span>
          {IDENTITIES.map((it) => (
            <button key={it.label} className={it.label === ident.label ? "primary" : ""} onClick={() => setIdent(it)}>
              {it.label}
            </button>
          ))}
        </div>
        <div className="row" style={{ margin: "8px 0" }}>
          <label className="muted">₹</label>
          <input type="number" value={amount} min={1}
                 onChange={(e) => setAmount(Number(e.target.value))} style={{ width: 90 }} />
          <button className="primary" onClick={pay} disabled={!keyId}>create order &amp; pay</button>
        </div>
        <div className="muted" style={{ fontSize: 12 }}>
          test card 4111 1111 1111 1111 · any future expiry / CVV · OTP 1111.
        </div>

        <div className="row" style={{ gap: 6, marginTop: 12, borderTop: "1px solid var(--border-2)", paddingTop: 10 }}>
          <span className="muted" style={{ fontSize: 12 }}>or seed a scenario:</span>
          <button onClick={() => seed("shared_card")}>shared card ×4</button>
          <button onClick={() => seed("carding")}>carding ×4</button>
        </div>

        <div style={{ marginTop: 10 }}>
          {log.map((l, i) => <div key={i} className="mono" style={{ fontSize: 12, color: "var(--muted)" }}>{l}</div>)}
        </div>
      </div>

      <div className="grid" style={{ gap: 14 }}>
        <div className="panel">
          <h2>Rings from live payments</h2>
          {rings.length === 0 && <div className="muted" style={{ fontSize: 13 }}>none yet — make ≥2 coordinated payments</div>}
          {rings.map((r) => (
            <div key={r.id} className="panel ring-card" style={{ margin: "8px 0" }}>
              <div className="ring-head">
                <strong>#{r.id} · {r.kind}{r.n_merchants > 1 && <span className="pill" style={{ marginLeft: 6 }}>{r.n_merchants} merchants</span>}</strong>
                <span className={"risk " + (r.score_mean > 0.5 ? "hi" : "mid")}>{r.score_mean.toFixed(2)}</span>
              </div>
              <div className="muted" style={{ fontSize: 12 }}>
                {r.size} payments · {r.distinct_members} identities · {r.distinct_cards} card{r.distinct_cards === 1 ? "" : "s"}
                {r.flagged ? <span className="pill won" style={{ marginLeft: 6 }}>flagged</span>
                           : <span className="pill" style={{ marginLeft: 6 }}>watch</span>}
              </div>
              <div className="key">{r.key}</div>
            </div>
          ))}
        </div>

        <div className="panel">
          <h2>Live payments in the store</h2>
          <table>
            <thead><tr><th>payment</th><th>risk</th><th>₹</th><th>identity</th><th>card</th><th>merchant</th><th>disputed</th></tr></thead>
            <tbody>
              {live.map((t) => (
                <tr key={t.id} className={t.flagged ? "flagged" : ""}>
                  <td className="mono">{t.id.slice(0, 18)}</td>
                  <td className={"risk " + (t.score > 0.5 ? "hi" : "mid")}>{t.score.toFixed(2)}</td>
                  <td>{t.amount.toFixed(0)}</td>
                  <td className="mono muted">{t.uid ?? "—"}</td>
                  <td className="mono muted">{t.card_id?.slice(0, 22) ?? "—"}</td>
                  <td className="muted">{t.merchant ?? "—"}</td>
                  <td>{t.disputed ? <span className="pill lost">yes</span> : "—"}</td>
                </tr>
              ))}
              {live.length === 0 && <tr><td colSpan={7} className="muted">no live payments yet</td></tr>}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
