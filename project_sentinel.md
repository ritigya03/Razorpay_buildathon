# Project Sentinel: The Privacy Adapter for Razorpay Vulcan

**Track:** Track 02 — AI Risk Manager
**Buildathon:** Razorpay AI Buildathon (applications close 5 September 2026)
**Tagline:** *"We don't replace Vulcan. We unlock it for the privacy-first world."*

> **This is the concept doc / original vision.** For what is actually built and
> measured, see [`README.md`](README.md) and the phase write-ups in `report/`.
> Key deviations from this doc, all recorded in [`DEVLOG.md`](DEVLOG.md):
> the **Shadow Engine** (§4.2) was not built — the cross-merchant latency bridge
> is instead the Phase-8 federated protocol (`train/fl_rings.py`,
> `backend/app/fl_live.py`); the **agent** (§4.3) runs on Google Gemini's free
> tier, not the Claude Agent SDK / Vertex, but is the same supervised
> tool-calling shape; the graded scorer is centralised LightGBM (federated
> learning of the *scorer* is Phase 4, an MLP; Phase 8 federates the *ring
> detection layer* with the LightGBM score held fixed). The core claim —
> *cross-merchant ring detection without pooling raw data, with DP* — is built
> and measured in [`report/PHASE8.md`](report/PHASE8.md): federated == centralised
> with no DP (F1 0.67 both), an honest DP cost below ε≈16, and a poisoned
> merchant caught and recovered.

---

## 1. Executive Summary

On 18 August 2026, Razorpay launched **Vulcan** — India's first AI Payments Foundation Model. Trained on ~4 billion payments and 3 trillion data points, using ~3,000 signals per transaction, it already does network-level, cross-merchant fraud detection in production (live with Blinkit, redBus, Bachatt), claiming an 8x improvement in international card fraud detection and 5x more fraudulent/disputed transactions identified.

Vulcan works by **centralizing raw transaction data** from participating merchants into one proprietary model. That centralization is a hard blocker for a specific, valuable segment:

- **Enterprise competitors** (e.g. two rival quick-commerce players) who will never pool raw data with each other.
- **RBI-regulated entities** (banks, NBFCs) bound by data-sovereignty norms.
- **Global enterprises** bound by GDPR/CCPA-style constraints.

**Project Sentinel** is a privacy-preserving adapter for this segment. Using **Federated Learning (FL)** and **Differential Privacy (DP)**, it aims to deliver the same class of network-level fraud signal Vulcan provides — cross-merchant pattern detection, e.g. shared IP clusters or device fingerprints behind a fraud ring — without merchants ever handing over raw transaction data to a central model.

Sentinel does not claim to replace Vulcan. It complements it: Vulcan is the real-time brain; Sentinel is the privacy-preserving layer for merchants who can't or won't feed Vulcan raw data.

---

## 2. How We Arrived at This Idea

This section exists so the reasoning is on record — both for the pitch narrative and so future-you remembers why each decision was made.

1. **Started broad.** Considered all five buildathon tracks. Ruled out Finance Controller (least aligned to background, thin gap) and Open Track (too unconstrained for a 5-day sprint with no existing thesis).
2. **Researched what Razorpay had already shipped**, to avoid pitching a worse version of an existing product. Found via `razorpay.com/sprint/26`, `agent-studio`, and `agentic-business-banking` that Razorpay already has named, live/beta agents for: Dispute Responder (chargebacks), Subscription Recovery, Abandoned Cart Conversion, RTO Shield/Insights, Settlement Insights, Cashflow Forecaster, Receivables Agent, Payout Agent, Bookkeeping Agent, Reporting Agent, with Tax Payments and Reimbursement agents "upcoming." This eliminated most of the "obvious" ideas in Tracks 02, 03, and 04.
3. **Identified the one visible gap:** nowhere across Razorpay's public agent roadmap is there a named pre-transaction fraud or abuse-ring detection agent — chargebacks (post-fraud) and RTO (returns) are covered, but proactive fraud detection wasn't, as of the initial research pass.
4. **Connected the gap to personal strengths.** The user is doing a final-year project (CardioTrust FL) on blockchain-assisted, differentially private federated learning for healthcare — i.e. privacy-preserving collaborative ML across data silos. Cross-merchant fraud detection without merchants sharing raw data is structurally the same problem. This became the anchor: an idea only this specific applicant could credibly pitch.
5. **Took inspiration from a hackathon-winning submission ("Cassandra")** — an agent that watches other agents for failures, self-audits, and demos live rather than via slides. Borrowed the pattern (not the tech): live before/after demo, an agent that reports honestly on its own accuracy, and an explicit acknowledgment of a limitation rather than a polished claim of perfection.
6. **Drafted v1 ("the patched version")** addressing seven anticipated judge objections: merchant incentive, cold-start noise, false positives, FL/real-time latency mismatch, weak differential privacy, unverifiable audit trail, and "this isn't actually agentic."
7. **Critical correction during research:** discovered Vulcan itself was launched on 18 August 2026 and already performs cross-merchant fraud detection — invalidating v1's central claim that Razorpay is "blind" to cross-merchant fraud. Also found a live, external, already-public criticism (DQIndia) questioning Vulcan's data-privacy/merchant-isolation methodology at its current scale.
8. **Reframed v2 around that finding**: Sentinel stopped claiming to fill a blind spot and started claiming to extend Vulcan's benefit to a segment Vulcan's centralized design structurally can't reach — a stronger, more defensible, more mature pitch grounded in a real and very recent competitor/internal system rather than a strawman.
9. **Fact-checked a quote** attributed to "Razorpay's official blog" and found it was not present verbatim on the actual Vulcan landing page (`razorpay.com/foundation-model`) — it was a paraphrase of press coverage (Business Standard). Corrected to avoid misattributing a fabricated quote to Razorpay in a pitch aimed at Razorpay.
10. **Trimmed the technical scope** for 5-day solo buildability: dropped real Sepolia/Ethereum blockchain integration in favor of a computed-and-logged Merkle root, and dropped Redis in favor of an in-memory sliding window — same architectural story, far less infrastructure risk.

---

## 3. The Problem, Precisely

| What Vulcan does (as launched, publicly) | The gap Sentinel targets |
|---|---|
| Spots fraud rings across thousands of merchants | Requires centralizing raw merchant data into one model |
| Processes ~3,000 signals per transaction | Not publicly disclosed to use Differential Privacy or Federated Learning |
| Live in production with Blinkit, redBus, Bachatt | Banks, regulated entities, and mutually competitive enterprise merchants have structural reasons not to pool raw data into a shared system |

An industry outlet (DQIndia) has already publicly raised the underlying question — noting Razorpay hasn't disclosed baseline detection rates, testing methodology, or how data privacy and merchant isolation are handled at that scale. Sentinel is framed as one credible answer to that open question, not as a claim that Razorpay's engineers haven't thought about it.

---

## 4. The Solution: How Sentinel Works

Sentinel is an **agentic federated learning system** that detects cross-merchant fraud rings while raw transaction data never leaves a merchant's own environment.

### 4.1 The FL Pipeline

1. **Local training** — each merchant trains a small anomaly-detection model on its own transaction data, using **Focal Loss** (to handle the rare-fraud class imbalance) and **DP-SGD via Opacus** (to add calibrated privacy noise, target ε ≈ 1.5).
2. **Commit phase** — merchants send a SHA-256 hash of their model weights to the central aggregator before revealing them, preventing "last-mover" attacks where a bad actor waits to see others' updates before submitting a poisoned one.
3. **Reveal & verify** — once all hashes are collected, merchants send the actual weights; the server checks each against its earlier commitment.
4. **Anomaly filtering** — the server computes each update's cosine distance from the median update; updates deviating sharply (threshold > 0.3) are rejected as likely poisoned gradients.
5. **Balance-weighted aggregation** — instead of standard FedAvg (weighted by transaction volume), Sentinel weights merchants by their minority-class (fraud) count, so a high-volume, low-fraud merchant can't dilute the signal.
6. **Global insight** — the aggregated model surfaces overlapping patterns (shared IP clusters, device fingerprints) across merchants, revealing a ring no single merchant could see alone.

### 4.2 The Shadow Engine (Latency Bridge)

Full FL convergence takes hours; Vulcan-class systems operate in milliseconds. The Shadow Engine bridges this:

- A 5-minute sliding window (built with Python's `collections.deque` — no Redis dependency) watches for the same IP cluster or device fingerprint reported by 2+ merchants.
- On a match, it immediately publishes a temporary **Shadow Rule** to a mocked feature-store endpoint (explicitly presented as a mock in the demo — not a claim of real Vulcan access), bypassing the full FL convergence cycle.
- The full FL model runs on its own cadence to refine long-term patterns and recalibrate Shadow Rules.

### 4.3 The Agent Layer

The pipeline is wrapped in a **LangGraph supervisor agent** with three tools:

- `run_detection()` — triggers the FL/Shadow Engine pipeline and returns findings.
- `generate_forensic_report()` — an LLM (Gemini via Vertex AI) turns technical output into a plain-language report, e.g. *"5 merchants share Device X and IP Y — consistent with a coordinated attack."*
- `suggest_mitigation()` — proposes next steps and can draft an escalation note.

This turns Sentinel from a script into something a human risk manager can converse with: *"What happened this morning?"* → agent queries the model → drafts a report → proposes a Shadow Rule.

---

## 5. Answers to Anticipated Objections

| Objection | Answer |
|---|---|
| **Why would a merchant participate?** | Target the top ~1,000 enterprise merchants who actively lose money to fraud, not all SMBs. Offer a "Security Shield" badge on checkout (trust/conversion benefit) and a discount on chargeback-protection fees. |
| **What about merchants with near-zero fraud (cold start / noise)?** | Active sampling: only request updates from merchants with at least 5 fraud attempts in the last 7 days; FL rounds are event-driven, not time-driven. |
| **Won't this create false positives (e.g. blocking a shared corporate IP)?** | Sentinel is advisory-only. It outputs a 0–100 confidence score; above 95%, it escalates (a small risk-score delta) rather than auto-blocking. A human reviews a forensic report before any large-scale action. |
| **FL is slow, real-time fraud engines aren't — how do they meet?** | The Shadow Engine's 5-minute sliding window pushes a temporary rule the moment 2+ merchants report the same signal, without waiting for full FL convergence. |
| **Is ε = 1.5 differential privacy strong enough?** | We show an accuracy-vs-privacy curve: at ε ≈ 1.5, utility drops by roughly 2% in exchange for privacy in the range typically considered defensible for GDPR/RBI-grade sensitivity. This trade-off is stated openly rather than hidden. |
| **How do we know the central aggregator isn't tampering with commits?** | All commit hashes are aggregated into a Merkle tree; the root is computed and logged (printed) at the end of each round. In production this could be anchored to a public timestamping service — for the demo, this is stated explicitly as a proposal, not a live integration. |
| **Where's the "agent" — isn't this just a script?** | The LangGraph supervisor layer (Section 4.3) is what turns the pipeline into something an autonomous agent does and reports on, rather than a one-off batch job. |

---

## 6. Technical Architecture

### 6.1 Stack (trimmed for 5-day solo buildability)

| Component | Technology | Purpose |
|---|---|---|
| Language | Python 3.12 | Core logic |
| ML / FL core | PyTorch + Flower | Local training & aggregation |
| Privacy | Opacus | DP-SGD, target ε ≈ 1.5 |
| Agent framework | LangGraph | Supervisor agent & tools |
| Shadow Engine | `collections.deque` | 5-minute sliding window (no Redis) |
| Audit | `hashlib` Merkle root | Computed and logged, not on-chain |
| UI | Streamlit | Demo cockpit |
| LLM | Gemini via Vertex AI | Forensic report generation |

### 6.2 Data Flow

```
Merchant Clients (simulated)         Sentinel Central Aggregator (FastAPI)
  |  commit (hash)          ------->
  |  reveal (weights)       <-------
                                       |
        +------------------------------+------------------------------+
        |                              |                              |
        v                              v                              v
  Hash verify /                  Shadow Engine                 LangGraph Agent
  cosine filter /                (deque, 5-min window)         - run_detection()
  balance-weighted avg /         -> mock Vulcan endpoint       - generate_report()
  Merkle root (logged)                                          - suggest_mitigation()
```

---

## 7. 5-Day Build Plan

The build order below prioritizes an early, ugly, end-to-end pipeline over a polished-but-untested one — if Day 1 produces a working (even if crude) pipeline with real numbers, that's the fallback submission no matter what happens afterward.

| Day | Focus | Deliverable |
|---|---|---|
| **Day 1** | End-to-end skeleton | A stubbed pipeline — simplified aggregation math (no real Flower rounds yet) — running on synthetic multi-merchant data with injected fraud rings, `random.seed(42)` for reproducibility, producing an initial precision/recall number. This is the insurance policy: something submittable exists from Day 1 onward. |
| **Day 2** | Real FL core | Swap in real Flower + Opacus DP-SGD + Focal Loss local training; commit/reveal via SHA-256; cosine-distance filtering; balance-weighted aggregation. Run 3 rounds, print F1. |
| **Day 3** | Shadow Engine + early UI | `shadow_engine.py` using `deque`; alert logic for same IP cluster across 2+ merchants within 5 minutes, printing a "Shadow Rule pushed" event. Start a bare-bones UI (even plain terminal output) now, not on Day 5. |
| **Day 4** | Agent layer | LangGraph setup with the three tools (`run_detection`, `generate_forensic_report`, `suggest_mitigation`); test the graph end to end. |
| **Day 5** | Polish + record only | Finish the Streamlit split-screen cockpit; run the deterministic replay showing the Shadow Rule catching a ring in real time; record the pitch video. No new logic is written on Day 5 — only polishing and recording. |

**Fallback if the FL core isn't producing sane numbers by end of Day 2:** drop the "federated" runtime and demo centralized ring-detection with DP noise added, presenting the full federated architecture as a diagram/roadmap rather than running code. A smaller thing that works beats a bigger thing that's still broken on demo day.

---

## 8. Demo Script (~3 minutes)

| Time | Visual | Narration |
|---|---|---|
| 0:00–0:30 | Merchant A & B dashboards, each showing an isolated flagged transaction, neither connecting the two | "Vulcan proved cross-merchant intelligence works — 8x better fraud detection. But it relies on centralizing data. What about enterprises who won't share data with competitors?" |
| 0:30–1:05 | Aggregator screen: "Ring detected across 3 merchants," commitment hashes visible | "Sentinel uses federated learning to detect overlapping fingerprints across merchants — without ever seeing raw data. Cryptographic commitments prevent last-mover attacks." |
| 1:05–2:00 | Replay the same fraudulent transaction; Shadow Rule fires in real time against the mocked endpoint | "This is the latency bridge. While the full model trains, the Shadow Engine catches the pattern in five minutes and pushes a rule to a mock Vulcan-style feature store — stopping the fraudster mid-attack." |
| 2:00–2:45 | Agent chat: "Report on today's rings" → forensic report generated live | "This isn't just a number — the agent explains the why, and drafts an escalation note for the merchant." |
| 2:45–3:00 | Terminal shows the logged Merkle root; privacy-budget indicator at ε ≈ 1.5 | "A verifiable audit trail and enterprise-grade privacy — Sentinel is a path to extend Vulcan-class protection to merchants who can't pool raw data." |

---

## 9. Positioning Notes (Do / Don't for the Writeup)

- **Do** frame Sentinel as complementary to Vulcan, not a replacement or a claim that Razorpay "missed" something.
- **Do** attribute the "proprietary architecture and data" characterization to press coverage (e.g. Business Standard), not to Razorpay's own blog — the exact quote used in an earlier draft was not found verbatim on Razorpay's Vulcan landing page.
- **Do** say "not publicly disclosed to use DP/FL," not "doesn't use DP/FL" — the difference matters if a Razorpay engineer reads this.
- **Do** narrate any Vulcan integration point as explicitly mocked in the demo.
- **Don't** use invented precise monetization figures (e.g. a specific dollar valuation) without a way to defend them if asked.
- **Don't** let the last build day include new, untested logic — Day 5 is polish and recording only.

---

## 10. Why This Fits the Applicant

This idea is deliberately chosen to be hard for another applicant to credibly replicate: it requires genuine familiarity with differentially private federated learning across data silos, which is the exact subject of the applicant's ongoing final-year project (CardioTrust FL, a blockchain-assisted differentially private federated learning system for secure healthcare analytics, supervised by Prof. Ashish Kumar Dwivedi at LNMIIT). Sentinel applies the same core technique — collaborative learning across parties who cannot share raw data — to a different domain (payments fraud) where Razorpay has just shown, publicly and very recently, that the underlying problem (cross-merchant fraud signal) is valuable enough to build a foundation model for.
