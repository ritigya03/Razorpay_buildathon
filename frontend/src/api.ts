export interface ReplayStatus {
  loaded: boolean; running: boolean; ingested: number; total: number;
  progress: number; virtual_day: number; days_per_sec: number; per_txn_threshold: number;
}

export interface Stats {
  replay: ReplayStatus;
  transactions: number; fraud_transactions: number;
  flagged: number; flagged_true_positive: number;
  live_precision_so_far: number | null;
  rings_total: number; rings_flagged: number;
  disputes: number; disputes_flagged_in_advance: number; avg_lead_time_hours: number;
  model_metrics: {
    pr_auc_test?: number; roc_auc_test?: number;
    operating_point?: { precision: number; recall: number; total_expected_cost: number; do_nothing_cost: number };
  };
}

export interface Txn {
  id: string; source: string; ts: number; amount: number;
  email_domain: string | null; card_id: string | null; device_id: string | null;
  uid: string | null; merchant: string | null;
  score: number; scorer: string; flagged: boolean;
  ring_id: number | null; is_fraud: number | null;
  disputed: boolean; dispute_outcome: string | null;
}

export interface Ring {
  id: number; kind: string; key: string; source: string; size: number;
  distinct_members: number; distinct_cards: number; n_merchants: number;
  score_mean: number; score_max: number; amount_total: number;
  first_ts: number; last_ts: number; flagged: boolean;
  n_fraud: number; n_disputed: number;
}

export interface Dispute {
  id: string; payment_id: string; source: string; phase: string | null;
  reason_code: string | null; amount: number; status: string;
  lead_time_hours: number | null; was_flagged: boolean; ring_id: number | null;
}

export interface Alert {
  id: number; ts: number; kind: string; ring_id: number | null;
  txn_id: string | null; summary: string; score: number;
}

export interface AgentHealth { ok: boolean; model: string | null; error: string | null; }
export interface AgentToolCall { tool: string; args: Record<string, unknown>; summary: string; }
export interface AgentReply { reply: string; tool_calls: AgentToolCall[]; model: string; }

const j = async (r: Response) => {
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
};

export const api = {
  health: () => fetch("/api/health").then(j),
  stats: (): Promise<Stats> => fetch("/api/stats").then(j),
  report: () => fetch("/api/report").then(j),
  flReport: () => fetch("/api/fl-report").then(j),
  flRingReport: () => fetch("/api/fl-ring-report").then(j),
  flDetectLive: (epsilon?: number | null) =>
    fetch("/api/fl/detect-live", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ epsilon: epsilon ?? null }),
    }).then(j),
  transactions: (q = "?limit=60"): Promise<Txn[]> => fetch(`/api/transactions${q}`).then(j),
  rings: (q = "?flagged=true&limit=40"): Promise<Ring[]> => fetch(`/api/rings${q}`).then(j),
  ring: (id: number): Promise<{ ring: Ring; members: Txn[] }> => fetch(`/api/rings/${id}`).then(j),
  disputes: (): Promise<Dispute[]> => fetch("/api/disputes").then(j),
  alerts: (q = "?limit=40"): Promise<Alert[]> => fetch(`/api/alerts${q}`).then(j),
  replay: (action: "start" | "pause" | "reset") =>
    fetch(`/api/replay/${action}`, { method: "POST" }).then(j),
  simulateDispute: (txn_id?: string) =>
    fetch("/api/simulate/dispute", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify(txn_id ? { txn_id } : {}),
    }).then(j),
  createOrder: (amount_paise: number, notes?: Record<string, string>) =>
    fetch("/api/orders", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ amount_paise, receipt: "sentinel_demo", notes }),
    }).then(j),
  verify: (body: Record<string, unknown>) =>
    fetch("/api/verify", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    }).then(j),
  demoScenario: (kind: "shared_card" | "carding", size = 4) =>
    fetch("/api/demo/scenario", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ kind, size }),
    }).then(j),

  agentHealth: (): Promise<AgentHealth> => fetch("/api/agent/health").then(j),
  agentChat: (message: string, session_id = "dashboard"): Promise<AgentReply> =>
    fetch("/api/agent/chat", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ message, session_id }),
    }).then(j),
  agentReset: (session_id = "dashboard") =>
    fetch("/api/agent/reset", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ session_id }),
    }).then(j),
};
