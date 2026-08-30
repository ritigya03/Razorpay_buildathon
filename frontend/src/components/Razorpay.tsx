import { useEffect, useState } from "react";
import { api, type Txn } from "../api";

declare global {
  interface Window { Razorpay?: any; }
}

export function RazorpayPanel({ keyId }: { keyId: string | null }) {
  const [amount, setAmount] = useState(500);
  const [log, setLog] = useState<string[]>([]);
  const [live, setLive] = useState<Txn[]>([]);
  const add = (s: string) => setLog((l) => [new Date().toLocaleTimeString() + "  " + s, ...l].slice(0, 12));

  useEffect(() => {
    const t = setInterval(
      () => api.transactions("?source=razorpay&limit=20").then(setLive).catch(() => {}), 2500);
    return () => clearInterval(t);
  }, []);

  const pay = async () => {
    try {
      const order = await api.createOrder(amount * 100);
      add(`order ${order.id} created (₹${amount})`);
      if (!window.Razorpay) { add("checkout.js not loaded"); return; }
      const rzp = new window.Razorpay({
        key: keyId, order_id: order.id, amount: order.amount, currency: "INR",
        name: "Sentinel Demo Merchant", description: "test payment",
        handler: (resp: any) => add(`paid — payment ${resp.razorpay_payment_id}`),
        modal: { ondismiss: () => add("checkout dismissed") },
        theme: { color: "#5b8def" },
      });
      rzp.open();
    } catch (e) {
      add("error: " + String(e));
    }
  };

  return (
    <div className="grid cols-2" style={{ alignItems: "start" }}>
      <div className="panel">
        <h2>Razorpay test-mode payment</h2>
        {!keyId && <div style={{ color: "var(--warn)" }}>Razorpay keys not configured on the backend.</div>}
        <div className="row" style={{ margin: "8px 0" }}>
          <label className="muted">₹</label>
          <input type="number" value={amount} min={1}
                 onChange={(e) => setAmount(Number(e.target.value))}
                 style={{ width: 100, background: "var(--panel-2)", color: "var(--text)", border: "1px solid var(--border)", borderRadius: 6, padding: "6px 8px" }} />
          <button className="primary" onClick={pay} disabled={!keyId}>create order &amp; pay</button>
        </div>
        <div className="muted" style={{ fontSize: 12 }}>
          test card 4111 1111 1111 1111 · any future expiry / CVV · OTP 1111. The captured payment
          arrives via webhook and is scored by rules (live features), then appears below.
        </div>
        <div style={{ marginTop: 10 }}>
          {log.map((l, i) => <div key={i} className="mono" style={{ fontSize: 12, color: "var(--muted)" }}>{l}</div>)}
        </div>
      </div>

      <div className="panel">
        <h2>Live Razorpay payments in the store</h2>
        <table>
          <thead><tr><th>payment</th><th>risk</th><th>amount</th><th>email</th><th>scorer</th><th>disputed</th></tr></thead>
          <tbody>
            {live.map((t) => (
              <tr key={t.id} className={t.flagged ? "flagged" : ""}>
                <td className="mono">{t.id}</td>
                <td className={"risk " + (t.score > 0.5 ? "hi" : "mid")}>{t.score.toFixed(2)}</td>
                <td>{t.amount.toFixed(0)}</td>
                <td className="muted">{t.email_domain ?? "—"}</td>
                <td><span className={"pill " + t.scorer}>{t.scorer}</span></td>
                <td>{t.disputed ? <span className="pill lost">yes</span> : "—"}</td>
              </tr>
            ))}
            {live.length === 0 && <tr><td colSpan={6} className="muted">no live payments yet</td></tr>}
          </tbody>
        </table>
      </div>
    </div>
  );
}
