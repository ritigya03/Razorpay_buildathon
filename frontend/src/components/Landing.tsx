import { useCountUp, useReveal } from "../hooks";

function Metric({ value, k, suffix = "", decimals = 0 }: {
  value: number; k: string; suffix?: string; decimals?: number;
}) {
  const v = useCountUp(value, { decimals, duration: 1400 });
  return (
    <div className="m">
      <div className="n">{v}{suffix}</div>
      <div className="k">{k}</div>
    </div>
  );
}

function Diagram() {
  const around = [
    [70, 60], [70, 180], [180, 30], [180, 210], [300, 70], [300, 170],
  ];
  const fraud = new Set([1, 3, 5]);
  return (
    <div className="diagram">
      <svg viewBox="0 0 560 240" role="img" aria-label="one shared device linking many cards">
        <defs>
          <radialGradient id="g" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#4de1f2" />
            <stop offset="100%" stopColor="#3395ff" />
          </radialGradient>
        </defs>
        {around.map(([x, y], i) => (
          <line key={i} className="edge" x1={430} y1={120} x2={x} y2={y} />
        ))}
        <circle className="ping" cx={430} cy={120} r={30} />
        <circle className="ping" cx={430} cy={120} r={30} style={{ animationDelay: "1.2s" }} />
        <circle className="hub" cx={430} cy={120} r={26} />
        <text x={430} y={124} textAnchor="middle" fontSize="10" fill="#04102b" fontWeight="700">device</text>
        {around.map(([x, y], i) => (
          <g key={i}>
            <circle className={"node" + (fraud.has(i) ? " fraud" : "")} cx={x} cy={y} r={13} />
            <text x={x} y={y + 3} textAnchor="middle" fontSize="9" fill="#9aa4d6">card</text>
          </g>
        ))}
      </svg>
    </div>
  );
}

export function Landing({ report, onEnter }: { report: any; onEnter: () => void }) {
  const m = report?.metrics;
  const dev = report?.ring_metrics?.rings?.device;
  const op = m?.operating_point?.test;
  const prauc = m?.pr_auc?.test ?? 0.546;
  const recall = op ? Math.round(op.recall * 100) : 84;
  const txnAlerts = op ? op.fp + op.tp : 18001;
  const ringAlerts = dev
    ? dev.rings_flagged + (report?.ring_metrics?.rings?.address?.rings_flagged ?? 0)
    : 132;
  const reduction = Math.round(txnAlerts / Math.max(ringAlerts, 1));
  const devPR = dev ? dev.ring_precision : 0.75;

  const s1 = useReveal<HTMLElement>();
  const s2 = useReveal<HTMLElement>();
  const s3 = useReveal<HTMLElement>();

  return (
    <div className="lp">
      <div className="grid-bg" />

      <nav className="nav">
        <span className="brand"><span className="mark">S</span> Project Sentinel</span>
        <button onClick={onEnter}>Open dashboard →</button>
      </nav>

      <div className="inner">
        <header className="hero reveal in">
          <h1>Stop the fraud ring <span className="g">before the chargeback.</span></h1>
          <p className="lede">
            Vulcan sees cross-merchant fraud by centralising raw transaction data. Sentinel reaches
            the same network-level signal without pooling it — and publishes the precision, recall
            and false-positive cost that a risk team needs before trusting a model.
          </p>
          <div className="cta">
            <button className="btn-grad" onClick={onEnter}>Open the live dashboard</button>
            <button className="ghost" onClick={() => document.getElementById("how")?.scrollIntoView()}>
              How it works
            </button>
          </div>
          <Diagram />
        </header>

        <div className="band">
          <Metric value={prauc} k="held-out PR-AUC" decimals={3} />
          <Metric value={devPR * 100} k="device-ring precision" suffix="%" />
          <Metric value={reduction} k="fewer alerts to review" suffix="×" />
          <Metric value={40} k="hours flagged before the dispute" suffix="h" />
        </div>

        <section className="sec reveal" id="how" ref={s1}>
          <h2>Three moving parts</h2>
          <p className="sub">
            A scorer for every transaction, an engine that groups the coordinated ones, and a loop
            that checks itself against real chargebacks.
          </p>
          <div className="cards">
            <div className="card">
              <div className="num">1</div>
              <h3>Transaction scorer</h3>
              <p>
                Gradient-boosted model on a strict forward-in-time split. Advisory 0–100 risk, never
                an auto-block. PR-AUC {prauc.toFixed(3)}, recall {recall}% at the cost-optimal threshold.
              </p>
            </div>
            <div className="card">
              <div className="num">2</div>
              <h3>Ring engine</h3>
              <p>
                Groups transactions that share a device fingerprint or shipping identity into
                coordinated rings — {txnAlerts.toLocaleString()} alerts become ~{ringAlerts} reviewable
                rings at {devPR.toFixed(2)} / {dev ? dev.ring_recall.toFixed(2) : "0.75"} precision/recall.
              </p>
            </div>
            <div className="card">
              <div className="num">3</div>
              <h3>Dispute loop</h3>
              <p>
                Razorpay chargebacks are the ground truth. Every dispute is matched back to the ring
                we flagged, with the lead time in hours.
              </p>
            </div>
          </div>
        </section>

        <section className="sec reveal" ref={s2}>
          <h2>We don't replace Vulcan. We unlock it for the privacy-first world.</h2>
          <div className="vs">
            <div className="col them">
              <h3>Centralised model</h3>
              <ul>
                <li>Raw transactions pooled into one model</li>
                <li>Data-sovereign and mutually-competitive merchants stay out</li>
                <li>Strong results, published without a baseline or false-positive rate</li>
              </ul>
            </div>
            <div className="col us">
              <h3>Project Sentinel</h3>
              <ul>
                <li>Raw data never leaves the merchant — only model updates are shared</li>
                <li>Differential privacy on those updates</li>
                <li>Precision, recall and a cost curve on a held-out set, in the open</li>
              </ul>
            </div>
          </div>
        </section>

        <section className="sec reveal" ref={s3}>
          <div className="strip">
            <h2>Measured in the open</h2>
            <p className="sub">
              Every number is on the final 15% of the timeline, split by date and never shuffled.
              The threshold is chosen on validation and applied once. The cockpit replays that
              window live, with real Razorpay test-mode payments alongside.
            </p>
            <button className="btn-grad" onClick={onEnter}>Enter the cockpit →</button>
          </div>
        </section>
      </div>

      <div className="foot">Project Sentinel · Razorpay AI Buildathon</div>
    </div>
  );
}
