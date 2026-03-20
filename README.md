# AgentApproved

**EU AI Act compliance evidence for AI agents.** One-line SDK integration captures every agent action as a tamper-proof, Ed25519-signed audit trail and maps it to specific regulatory requirements.

```
pip install agentapproved
```

## Quick Start

```python
from agentapproved import AgentApprovedHandler, assess_compliance, generate_evidence_packet
from agentapproved.hasher import get_public_key_hex

# 1. Attach to your LangChain agent
handler = AgentApprovedHandler(agent_id="my-agent", data_dir="./evidence")
agent = create_agent(..., callbacks=[handler])
agent.invoke({"input": "What is the return policy?"})

# 2. Record human oversight (EU AI Act Art 12(2)(d))
handler.record_oversight(reviewer_id="jane", decision="approved", reason="Accurate")
handler.end_session()

# 3. Check compliance score
report = assess_compliance(handler.get_events())
print(f"EU AI Act Article 12 compliance: {report.overall_score}%")
for gap in report.gaps:
    print(f"  {gap.article}: {gap.remediation}")

# 4. Export evidence packet for auditors
generate_evidence_packet(
    handler.get_events(),
    "./audit-packet",
    organisation="Acme Ltd",
    public_key_hex=get_public_key_hex(handler._public_key),
)
```

## What You Get

```
EU AI Act Article 12 compliance: 100%
  ✓ Article 12(1)    — Automatic logging capability
  ✓ Article 12(2)(a) — Period of each use
  ✓ Article 12(2)(b) — Reference database
  ✓ Article 12(2)(c) — Input data leading to match
  ✓ Article 12(2)(d) — Human oversight verification
  ✓ Article 12(3)    — Post-market monitoring traceability
```

The evidence packet contains:
- **report.html** — auditor-readable compliance report with article-by-article assessment
- **evidence.json** — machine-readable event data with compliance mapping
- **integrity.json** — hash chain and Ed25519 signature verification proof

Every event is SHA-256 hash-chained and Ed25519 signed. Tampering with any event breaks the chain. An auditor can independently verify the entire audit trail.

## Features

- **One-line integration** — `callbacks=[handler]` on any LangChain agent
- **All LangChain events** — LLM calls, tool use, RAG retrieval, agent decisions, chat models
- **EU AI Act Article 12 mapping** — automatic compliance scoring with remediation guidance
- **Tamper-proof** — SHA-256 hash chain + Ed25519 signatures on every event
- **Human oversight capture** — `record_oversight()` satisfies Article 12(2)(d)
- **Self-contained evidence packets** — HTML report + JSON data + integrity proof
- **Local-first** — events persist to local JSON files, no cloud required
- **Never crashes your agent** — all errors swallowed, logging only

## Requirements

- Python 3.10+
- LangChain (`langchain-core >= 0.3.0`)

## Documentation

Full documentation at [agentapproved.ai/docs](https://agentapproved.ai/docs)

## License

MIT — see [LICENSE](LICENSE)
