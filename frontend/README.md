# Frontend (Phase 6) — React dashboard + landing

Vite + React + TypeScript. Recharts for the two curves; the ring graph and the
landing-page network background are hand-drawn (SVG / canvas). No CSS framework
or animation library — one `styles.css`.

**Theme:** Razorpay palette — navy `#02042B`, Prussian `#0C2651`, Razorpay blue
`#0D94FB` / `#3395FF`, cyan `#4DE1F2`, with a blue→cyan gradient for accents.
Fonts: Space Grotesk (headings) + Inter (body) via Google Fonts.

**Flow:** animated page loader (spinning rings + wordmark + a progress bar that
waits for `/api/health`, `/api/report`, `/api/stats`) → **landing page**
(animated node-network canvas, floating gradient orbs, moving grid, count-up
metric strip, scroll-reveal sections, "Vulcan vs Sentinel" framing) →
**dashboard**. Routing is hash-based (`#/app`); click the wordmark to go back.
Respects `prefers-reduced-motion`.

## Run

```bash
# backend first (repo root): make backend      -> http://localhost:8000
cd frontend
npm install
npm run dev            # http://localhost:5173  (proxies /api and /webhook to :8000)
```

`npm run build` type-checks (`tsc -b`) and bundles to `dist/`.

## Tabs

| Tab | Shows |
|---|---|
| **Overview** | replay progress + controls, live counts, live precision, model PR/ROC-AUC, ring alerts |
| **Feed** | live transaction stream — risk score, source (`model` replay / `rules` live), ring id, ground-truth label |
| **Rings** | flagged coordinated rings; click one → member table + SVG (nodes = transactions, red = fraud, colour = distinct card, green halo = disputed) |
| **Disputes** | dispute/chargeback list with lead time; "simulate a dispute" button; average-lead-time callout |
| **Metrics** | held-out precision–recall curve, expected-loss-vs-recall curve (cost-optimal point marked), ring-engine table |
| **Razorpay** | create a real test-mode order + open Razorpay Checkout; live payments (rules-scored) appear as they arrive |

All views poll the backend every 2–3 s.
