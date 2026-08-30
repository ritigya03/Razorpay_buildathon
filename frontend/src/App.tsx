import { useEffect, useState } from "react";
import { api, type Stats } from "./api";
import { Loader } from "./components/Loader";
import { Landing } from "./components/Landing";
import { Overview } from "./components/Overview";
import { Feed } from "./components/Feed";
import { Rings } from "./components/Rings";
import { Disputes } from "./components/Disputes";
import { Metrics } from "./components/Metrics";
import { RazorpayPanel } from "./components/Razorpay";

const TABS = ["Overview", "Feed", "Rings", "Disputes", "Metrics", "Razorpay"] as const;
type Tab = (typeof TABS)[number];

export default function App() {
  const [booted, setBooted] = useState(false);
  const [progress, setProgress] = useState(8);
  const [report, setReport] = useState<any>(null);
  const [health, setHealth] = useState<any>(null);

  const [route, setRoute] = useState(() => (location.hash === "#/app" ? "app" : "landing"));
  const [tab, setTab] = useState<Tab>("Overview");
  const [stats, setStats] = useState<Stats | null>(null);

  // boot sequence
  useEffect(() => {
    const creep = setInterval(() => setProgress((p) => Math.min(p + Math.random() * 9, 88)), 180);
    Promise.allSettled([
      api.health().then(setHealth),
      api.report().then(setReport),
      api.stats().then(setStats),
    ]).then(() => {
      clearInterval(creep);
      setProgress(100);
      setTimeout(() => setBooted(true), 550);
    });
    return () => clearInterval(creep);
  }, []);

  // hash routing
  useEffect(() => {
    const onHash = () => setRoute(location.hash === "#/app" ? "app" : "landing");
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  // live stats polling (only needed in the app)
  useEffect(() => {
    if (route !== "app") return;
    const t = setInterval(() => api.stats().then(setStats).catch(() => {}), 2000);
    return () => clearInterval(t);
  }, [route]);

  const enter = () => { location.hash = "#/app"; setRoute("app"); };
  const home = () => { location.hash = ""; setRoute("landing"); };

  return (
    <>
      {!booted && <Loader progress={progress} done={progress >= 100} />}

      {booted && route === "landing" && <Landing report={report} onEnter={enter} />}

      {booted && route === "app" && (
        <div className="app">
          <div className="topbar">
            <span className="brand" style={{ cursor: "pointer" }} onClick={home}>
              <span className="dot" /> Project <b>Sentinel</b>
            </span>
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
      )}
    </>
  );
}
