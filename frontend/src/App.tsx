import { useEffect, useState } from "react";
import { api, type Stats } from "./api";
import { Overview } from "./components/Overview";
import { Feed } from "./components/Feed";
import { Rings } from "./components/Rings";
import { Disputes } from "./components/Disputes";
import { Metrics } from "./components/Metrics";
import { RazorpayPanel } from "./components/Razorpay";

const TABS = ["Overview", "Feed", "Rings", "Disputes", "Metrics", "Razorpay"] as const;
type Tab = (typeof TABS)[number];

export default function App() {
  const [tab, setTab] = useState<Tab>("Overview");
  const [stats, setStats] = useState<Stats | null>(null);
  const [health, setHealth] = useState<any>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth({ ok: false }));
    const t = setInterval(() => api.stats().then(setStats).catch(() => {}), 2000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="app">
      <div className="topbar">
        <span className="brand">Project <span>Sentinel</span></span>
        <span className="tag">AI Risk Manager · coordinated-fraud rings</span>
        <span className={"tag " + (health?.model_loaded ? "live" : "off")}>
          model {health?.model_loaded ? "loaded" : "—"}
        </span>
        <span className={"tag " + (health?.razorpay_ready ? "live" : "off")}>
          razorpay {health?.razorpay_ready ? "connected" : "no keys"}
        </span>
      </div>

      <div className="tabs">
        {TABS.map((t) => (
          <button key={t} className={t === tab ? "active" : ""} onClick={() => setTab(t)}>{t}</button>
        ))}
      </div>

      {tab === "Overview" && <Overview stats={stats} />}
      {tab === "Feed" && <Feed />}
      {tab === "Rings" && <Rings />}
      {tab === "Disputes" && <Disputes stats={stats} />}
      {tab === "Metrics" && <Metrics />}
      {tab === "Razorpay" && <RazorpayPanel keyId={health?.razorpay_key_id ?? null} />}
    </div>
  );
}
