import { useEffect, useRef, useState } from "react";
import { api, type AgentHealth, type AgentToolCall } from "../api";

type Msg = {
  role: "user" | "agent";
  text: string;
  tools?: AgentToolCall[];
  error?: boolean;
};

const PROMPTS = [
  "Brief me — what's happening in the feed right now?",
  "List the worst coordinated rings and why they look coordinated.",
  "Give me a forensic report on the highest-risk ring.",
  "Draft an escalation note for that ring.",
  "Have we flagged the recent disputes before the chargebacks?",
];

// --- tiny, safe markdown (escape first, then inject only known tags) --------
function md(src: string): string {
  const esc = src
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const lines = esc.split("\n");
  const out: string[] = [];
  let i = 0;
  const inline = (s: string) =>
    s
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");

  while (i < lines.length) {
    const ln = lines[i];
    if (/^\s*#{1,4}\s+/.test(ln)) {
      out.push(`<div class="md-h">${inline(ln.replace(/^\s*#{1,4}\s+/, ""))}</div>`);
      i++;
    } else if (/^\s*(-{3,}|\*{3,})\s*$/.test(ln)) {
      out.push("<hr/>");
      i++;
    } else if (/^\s*\|.*\|\s*$/.test(ln)) {
      const rows: string[] = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) {
        rows.push(lines[i]);
        i++;
      }
      const cells = (r: string) =>
        r.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const body = rows.filter((r) => !/^\s*\|[\s:|-]+\|\s*$/.test(r));
      const head = body.length ? cells(body[0]) : [];
      out.push(
        `<table class="md-t"><thead><tr>${head
          .map((c) => `<th>${inline(c)}</th>`)
          .join("")}</tr></thead><tbody>${body
          .slice(1)
          .map(
            (r) =>
              `<tr>${cells(r).map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`
          )
          .join("")}</tbody></table>`
      );
    } else if (/^\s*[-*]\s+/.test(ln)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(`<li>${inline(lines[i].replace(/^\s*[-*]\s+/, ""))}</li>`);
        i++;
      }
      out.push(`<ul>${items.join("")}</ul>`);
    } else if (ln.trim() === "") {
      out.push("");
      i++;
    } else {
      out.push(`<p>${inline(ln)}</p>`);
      i++;
    }
  }
  return out.join("\n");
}

export function Agent() {
  const [health, setHealth] = useState<AgentHealth | null>(null);
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.agentHealth().then(setHealth).catch(() =>
      setHealth({ ok: false, model: null, error: "backend unreachable" })
    );
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [msgs, busy]);

  const send = async (text: string) => {
    text = text.trim();
    if (!text || busy) return;
    setInput("");
    setMsgs((m) => [...m, { role: "user", text }]);
    setBusy(true);
    try {
      const r = await api.agentChat(text);
      setMsgs((m) => [...m, { role: "agent", text: r.reply, tools: r.tool_calls }]);
    } catch (e) {
      const raw = e instanceof Error ? e.message : String(e);
      const m = raw.match(/\{"detail":"(.*)"\}/);
      const detail = m ? m[1] : raw;
      setMsgs((mm) => [
        ...mm,
        {
          role: "agent",
          error: true,
          text: detail.startsWith("RATE_LIMIT:")
            ? "⏳ Gemini free-tier limit hit — wait ~30s and try again."
            : `⚠️ ${detail}`,
        },
      ]);
    } finally {
      setBusy(false);
    }
  };

  const reset = () => {
    api.agentReset().catch(() => {});
    setMsgs([]);
  };

  if (health && !health.ok)
    return (
      <div className="panel">
        <h2>Risk-analyst agent — Phase 5</h2>
        <div className="muted">
          Not configured. Add a free Google Gemini key to{" "}
          <code>backend/.env</code>:
          <pre className="key" style={{ marginTop: 8 }}>
SENTINEL_GEMINI_API_KEY=&lt;key from aistudio.google.com/apikey&gt;
          </pre>
          {health.error && <div style={{ marginTop: 6 }}>backend says: {health.error}</div>}
        </div>
      </div>
    );

  return (
    <div className="panel agent">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h2 style={{ margin: 0 }}>Risk-analyst agent</h2>
        <span className="row" style={{ gap: 8 }}>
          {health?.model && <span className="tag live">{health.model}</span>}
          <button onClick={reset} disabled={busy || !msgs.length}>reset</button>
        </span>
      </div>

      <div className="muted" style={{ fontSize: 12, margin: "6px 0 12px" }}>
        Conversational layer over the live event store. It has read-only tools
        (situation summary, flagged rings, ring detail, recent disputes), grounds
        every number in a tool call, and drafts forensic reports / escalation
        notes. Advisory only.
      </div>

      <div className="chat">
        {msgs.length === 0 && (
          <div className="chat-empty muted">
            Ask about the current feed, a specific ring, or the dispute loop.
          </div>
        )}
        {msgs.map((m, k) => (
          <div key={k} className={"bubble " + m.role + (m.error ? " err" : "")}>
            {m.role === "agent" && m.tools && m.tools.length > 0 && (
              <div className="toolrow">
                {m.tools.map((t, j) => (
                  <span key={j} className="toolchip" title={JSON.stringify(t.args)}>
                    {t.tool} · {t.summary}
                  </span>
                ))}
              </div>
            )}
            {m.role === "agent" ? (
              <div className="md" dangerouslySetInnerHTML={{ __html: md(m.text) }} />
            ) : (
              m.text
            )}
          </div>
        ))}
        {busy && (
          <div className="bubble agent">
            <span className="dots"><i /><i /><i /></span>
          </div>
        )}
        <div ref={endRef} />
      </div>

      {msgs.length === 0 && (
        <div className="prompts">
          {PROMPTS.map((p) => (
            <button key={p} className="promptchip" disabled={busy} onClick={() => send(p)}>
              {p}
            </button>
          ))}
        </div>
      )}

      <form
        className="row chat-in"
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
      >
        <input
          type="text"
          placeholder={busy ? "thinking…" : "Ask the analyst…"}
          value={input}
          disabled={busy}
          onChange={(e) => setInput(e.target.value)}
          style={{ flex: 1 }}
        />
        <button className="primary" type="submit" disabled={busy || !input.trim()}>
          send
        </button>
      </form>
    </div>
  );
}
