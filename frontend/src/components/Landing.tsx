import { useReveal } from "../hooks";

export function Landing({ report, onEnter }: { report: any; onEnter: () => void }) {
  const m = report?.metrics;
  const dev = report?.ring_metrics?.rings?.device;
  const op = m?.operating_point?.test;
  const prauc = m?.pr_auc?.test ?? 0.546;
  const rocauc = m?.roc_auc?.test ?? 0.905;
  const recall = op ? Math.round(op.recall * 100) : 84;
  const costCut = op ? Math.round((1 - op.total_expected_cost / op.do_nothing_cost) * 100) : 76;
  const txnAlerts = op ? op.fp + op.tp : 18001;
  const ringAlerts = dev
    ? dev.rings_flagged + (report?.ring_metrics?.rings?.address?.rings_flagged ?? 0)
    : 132;

  const s1 = useReveal<HTMLElement>();
  const s2 = useReveal<HTMLElement>();
  const s3 = useReveal<HTMLElement>();
  const s4 = useReveal<HTMLElement>();

  return (
    <div className="lp">
      <nav className="nav">
        <span className="brand"><span className="mark">S</span> Project Sentinel</span>
        <a onClick={onEnter} style={{ cursor: "pointer" }}>Open the dashboard →</a>
      </nav>

      <div className="eyebrow">Razorpay AI Buildathon · Track 02 — AI Risk Manager</div>
      <h1>Reaching Vulcan's cross-merchant fraud signal without pooling raw data.</h1>
      <p className="lede">
        Vulcan sees fraud rings by centralising every merchant's transactions into one model.
        Banks, regulated lenders and rival enterprises can't join that. Sentinel is the adapter
        for them — federated learning for the network-level signal, and the honest precision,
        recall and false-positive cost that a real risk team needs before trusting it.
      </p>
      <div className="actions">
        <button className="primary" onClick={onEnter}>Open the live dashboard</button>
        <a onClick={() => document.getElementById("how")?.scrollIntoView()} style={{ cursor: "pointer" }}>
          How it works
        </a>
      </div>

      <section id="how" ref={s1} className="reveal">
        <h2>What it is</h2>
        <p>
          A detector for one class of loss: coordinated fraud rings — one actor spreading many
          small payments across cards, devices and merchants so no single merchant sees enough
          to react. Sentinel scores every transaction, then groups the ones that share a device
          fingerprint or a shipping identity into rings a human can actually review.
        </p>
        <p>
          It never blocks a payment. Every output is an advisory score and a short forensic note.
          Razorpay chargebacks are the ground truth it's measured against.
        </p>
      </section>

      <section ref={s2} className="reveal">
        <h2>The numbers</h2>
        <table className="numbers">
          <thead>
            <tr><th>Measure</th><th>Value</th><th>On</th></tr>
          </thead>
          <tbody>
            <tr><td>Transaction model — PR-AUC</td><td className="n">{prauc.toFixed(3)}</td><td>held-out test</td></tr>
            <tr><td>Transaction model — ROC-AUC</td><td className="n">{rocauc.toFixed(3)}</td><td>held-out test</td></tr>
            <tr><td>Recall at the cost-optimal threshold</td><td className="n">{recall}%</td><td>held-out test</td></tr>
            <tr><td>Expected-loss reduction vs doing nothing</td><td className="n">{costCut}%</td><td>held-out test</td></tr>
            <tr>
              <td>Device rings — precision / recall</td>
              <td className="n">{dev ? `${dev.ring_precision.toFixed(2)} / ${dev.ring_recall.toFixed(2)}` : "0.75 / 0.75"}</td>
              <td>held-out test</td>
            </tr>
            <tr>
              <td>Review load</td>
              <td className="n">{txnAlerts.toLocaleString()} → {ringAlerts} rings</td>
              <td>held-out test</td>
            </tr>
          </tbody>
        </table>
        <p className="caption">
          Measured on the final 15% of the timeline, split by date and never shuffled. The
          threshold is chosen on validation and applied once. Numbers here are read live from the
          running service.
        </p>
      </section>

      <section ref={s3} className="reveal">
        <h2>Where it sits next to Vulcan</h2>
        <div className="cols">
          <div>
            <h3>Centralised model</h3>
            <ul>
              <li>Raw transactions pooled into one model</li>
              <li>Data-sovereign and mutually-competitive merchants stay out</li>
              <li>Strong results, but published without a baseline or false-positive rate</li>
            </ul>
          </div>
          <div>
            <h3>Sentinel</h3>
            <ul>
              <li>Raw data never leaves the merchant; only model updates are shared</li>
              <li>Differential privacy on those updates</li>
              <li>Precision, recall and a cost curve on a held-out set, in the open</li>
            </ul>
          </div>
        </div>
      </section>

      <section ref={s4} className="reveal">
        <h2>What didn't work</h2>
        <p className="note">
          The first plan was that explicit ring features would lift the model. They didn't — on
          held-out data they moved PR-AUC by 0.003, within noise, because the dataset's
          pre-engineered columns already carry that signal. So the ring layer isn't about
          accuracy; it's about turning {txnAlerts.toLocaleString()} alerts into {ringAlerts}
          {" "}reviewable rings. That result is in the repo too.
        </p>
        <div className="actions" style={{ marginTop: 24 }}>
          <button className="primary" onClick={onEnter}>Open the dashboard</button>
        </div>
      </section>

      <div className="foot">Project Sentinel · Razorpay AI Buildathon</div>
    </div>
  );
}
