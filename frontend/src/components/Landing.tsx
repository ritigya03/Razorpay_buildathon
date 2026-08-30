import { useCountUp, useReveal } from "../hooks";
import { NetworkCanvas } from "./NetworkCanvas";

function Metric({ n, k, suffix = "", decimals = 0 }: { n: number; k: string; suffix?: string; decimals?: number }) {
  const v = useCountUp(n, { decimals, duration: 1400 });
  return (
    <div className="m">
      <div className="n">{v}{suffix}</div>
      <div className="k">{k}</div>
    </div>
  );
}

export function Landing({ report, onEnter }: { report: any; onEnter: () => void }) {
  const dev = report?.ring_metrics?.rings?.device;
  const prauc = report?.metrics?.pr_auc?.test ?? 0.546;
  const op = report?.metrics?.operating_point?.test;
  const txnAlerts = op ? op.fp + op.tp : 18001;
  const ringAlerts = dev ? dev.rings_flagged + (report?.ring_metrics?.rings?.address?.rings_flagged ?? 0) : 132;
  const reduction = Math.round(txnAlerts / Math.max(ringAlerts, 1));

  const s1 = useReveal<HTMLDivElement>();
  const s2 = useReveal<HTMLDivElement>();
  const s3 = useReveal<HTMLDivElement>();

  return (
    <div className="landing">
      <NetworkCanvas />
      <div className="grid-bg" />
      <div className="orb a" /><div className="orb b" />

      <div className="wrap">
        <nav className="nav">
          <span className="brand"><span className="dot" /> Project <b>Sentinel</b></span>
          <span className="chip"><span className="dot" /> live cockpit running</span>
        </nav>

        <header className="hero reveal in">
          <span className="chip">Razorpay AI Buildathon · Track 02 — AI Risk Manager</span>
          <h1>
            Catch the <span className="g">fraud ring</span> before the chargeback does.
          </h1>
          <p>
            Vulcan centralises raw transaction data to see across merchants. Sentinel
            reaches the same network-level fraud signal <strong>without pooling raw data</strong> —
            and publishes the honest precision, recall and false-positive cost that come with it.
          </p>
          <div className="cta">
            <button className="primary" onClick={onEnter}>Open the live cockpit →</button>
            <button onClick={() => document.getElementById("how")?.scrollIntoView()}>How it works</button>
          </div>
        </header>

        <div className="metrics-strip">
          <Metric n={prauc} k="held-out PR-AUC" decimals={3} />
          <Metric n={dev ? dev.ring_precision * 100 : 75} k="device-ring precision" suffix="%" />
          <Metric n={reduction} k="fewer alerts to review" suffix="×" />
          <Metric n={40} k="hours flagged before the dispute" suffix="h" />
        </div>

        <section className="section" id="how">
          <div ref={s1} className="reveal">
            <h2 className="big">Three moving parts</h2>
            <div className="steps">
              <div className="panel s hover">
                <div className="num">1</div>
                <h3>Transaction scorer</h3>
                <p>A gradient-boosted model on a strict forward-in-time split. Advisory 0–100
                  risk, never an auto-block. PR-AUC {prauc.toFixed(3)}, recall{" "}
                  {op ? Math.round(op.recall * 100) : 84}% at the cost-optimal threshold.</p>
              </div>
              <div className="panel s hover">
                <div className="num">2</div>
                <h3>Ring engine</h3>
                <p>Groups transactions that share a device fingerprint or shipping identity into
                  coordinated rings — turning {txnAlerts.toLocaleString()} alerts into ~{ringAlerts}
                  {" "}reviewable rings at {dev ? dev.ring_precision.toFixed(2) : "0.75"} /
                  {" "}{dev ? dev.ring_recall.toFixed(2) : "0.75"} precision/recall.</p>
              </div>
              <div className="panel s hover">
                <div className="num">3</div>
                <h3>Dispute loop</h3>
                <p>Razorpay chargebacks are the ground truth. Every dispute is matched back to the
                  ring we flagged — with the lead time in hours.</p>
              </div>
            </div>
          </div>
        </section>

        <section className="section">
          <div ref={s2} className="reveal">
            <h2 className="big">We don't replace Vulcan. We unlock it for the privacy-first world.</h2>
            <div className="vs">
              <div className="panel col them">
                <h3>Centralised model</h3>
                <ul>
                  <li>Needs raw transaction data pooled into one model</li>
                  <li>Banks, NBFCs and rival enterprises structurally can't join</li>
                  <li>Metrics published without a baseline or false-positive rate</li>
                </ul>
              </div>
              <div className="panel col us">
                <h3>Project Sentinel</h3>
                <ul>
                  <li>Network-level ring signal, raw data never leaves the merchant</li>
                  <li>Federated learning + differential privacy as the bridge</li>
                  <li>Precision / recall / cost curve on a held-out test set — in the open</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        <section className="section">
          <div ref={s3} className="reveal" style={{ textAlign: "center" }}>
            <h2 className="big">See it running on six months of real fraud data</h2>
            <p className="muted" style={{ maxWidth: "52ch", margin: "10px auto 24px" }}>
              The cockpit replays the held-out split through the live model and ring engine, with
              real Razorpay test-mode payments flowing in alongside.
            </p>
            <button className="primary" onClick={onEnter} style={{ padding: "12px 24px", fontSize: 15 }}>
              Enter the cockpit →
            </button>
          </div>
        </section>

        <div className="foot">Project Sentinel · built for the Razorpay AI Buildathon</div>
      </div>
    </div>
  );
}
