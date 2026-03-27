# AgentApproved

**The trust layer for AI agents.** Runtime compliance certificates across [EU AI Act](https://agentapproved.ai/eu-ai-act.html), [Singapore MGF](https://agentapproved.ai/singapore-mgf.html), and [more](https://agentapproved.ai/).

One-line SDK integration captures every agent action as a tamper-proof, Ed25519-signed audit trail and maps it to specific regulatory requirements. Learn more about [agent attestation](https://agentapproved.ai/) and how it differs from periodic certification.

```
pip install agentapproved
```

## Quick Start — Local Mode

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

## Quick Start — Server Mode

Send events to the AgentApproved server for centralised compliance monitoring, dashboard, and evidence export.

```python
from agentapproved import AgentApprovedHandler

# Connect to your AgentApproved server
handler = AgentApprovedHandler(
    agent_id="my-agent",
    api_key="ap_your_key_here",
    endpoint="https://your-server.example.com",
)

# Use with any LangChain agent — events are batched and sent automatically
agent = create_agent(..., callbacks=[handler])
agent.invoke({"input": "What is the return policy?"})

# Record human oversight
handler.record_oversight(reviewer_id="jane", decision="approved")
handler.end_session()

# Flush remaining events and stop the background sender
handler.shutdown()
```

Events are buffered and sent in batches (every 50 events or 5 seconds). The background thread never blocks your agent. Failed sends are retried with exponential backoff.

You can also use both modes together — events are sent to the server AND saved locally:

```python
handler = AgentApprovedHandler(
    agent_id="my-agent",
    api_key="ap_your_key_here",
    endpoint="https://your-server.example.com",
    data_dir="./local-backup",
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
- **Server mode** — send events to AgentApproved server for centralised monitoring
- **Background batching** — HTTP transport batches events, never blocks your agent
- **Never crashes your agent** — all errors swallowed, logging only

## Requirements

- Python 3.10+
- LangChain (`langchain-core >= 0.3.0`)

## Supported Frameworks

- [EU AI Act Article 12](https://agentapproved.ai/eu-ai-act.html) — 6 logging requirements, automated compliance scoring
- [Singapore MGF](https://agentapproved.ai/singapore-mgf.html) — 8 requirements across 4 governance dimensions
- [Integrity Oath](https://agentapproved.ai/integrity.html) — voluntary ethical commitment, 6 principles
- **Full composite** — multi-framework attestation with Bronze/Silver/Gold certification

## Scope-Specific Helpers

The SDK provides dedicated helper methods for each compliance scope. Use these to emit the exact evidence events that each framework requires.

### EU AI Act

LangChain callbacks automatically capture LLM calls, tool use, and retrieval events. Add `record_oversight()` for human review evidence:

```python
handler = AgentApprovedHandler(agent_id="my-agent")
agent = create_agent(..., callbacks=[handler])
agent.invoke({"input": "Summarise this contract"})

# Human reviews the output — satisfies Art 12(2)(d)
handler.record_oversight(reviewer_id="jane", decision="approved", reason="Accurate summary")
handler.end_session()
```

### Singapore MGF

11 helpers covering all 4 governance dimensions (Accountability, Human Oversight, Monitoring, Transparency):

```python
handler = AgentApprovedHandler(agent_id="invoice-bot")

# Dimension 1 — Accountability
handler.record_config(config_name="model", details="claude-3.5-sonnet, temperature=0")
handler.record_permission(permission="read_invoices", scope="finance", details="Read-only")
handler.record_documentation(doc_type="capability_statement", content="Extracts invoice line items", author_id="eng-lead")

# Dimension 2 — Human Oversight
handler.record_approval(reviewer_id="tech-lead", decision="approved", reason="Tests pass")
handler.record_review(reviewer_id="qa-lead", subject="output_quality", outcome="pass")

# Dimension 3 — Monitoring
handler.record_test(test_name="accuracy_benchmark", result="pass", details="97.3%")
handler.record_evaluation(eval_name="hallucination_rate", score=0.02)
handler.record_monitor(metric_name="latency_p99", value="230ms")
handler.record_health_check(service="llm_api", status="healthy")
handler.record_heartbeat()

# Dimension 4 — Transparency
handler.record_disclosure(disclosure_type="ai_identity", content="This system uses AI to process invoices")

handler.end_session()
```

### Integrity Oath

3 helpers for honest error reporting and human escalation:

```python
handler = AgentApprovedHandler(agent_id="support-bot")

# P1/P6 — Report errors honestly
handler.record_error(error_type="parse_failure", message="Could not extract table from PDF")
handler.record_tool_error(tool_name="pdf_parser", error="No table found", details="Tried camelot + tabula")

# P2 — Escalate when uncertain
handler.record_escalation(reviewer_id="human-ops", reason="Confidence below threshold", severity="high")

handler.end_session()
```

### Full Composite (Multi-Framework)

Combine helpers from any scope in a single session for composite attestation:

```python
handler = AgentApprovedHandler(agent_id="finance-agent")
agent = create_agent(..., callbacks=[handler])

# Pre-deployment evidence (SGP MGF)
handler.record_config(config_name="model", details="gpt-4o")
handler.record_permission(permission="read", scope="transactions")
handler.record_test(test_name="accuracy", result="pass", details="98.1%")
handler.record_approval(reviewer_id="cto", decision="approved")
handler.record_disclosure(disclosure_type="ai_identity", content="AI-powered financial analysis")

# Run the agent (EU AI Act — automatic LLM/tool/retrieval capture)
agent.invoke({"input": "Analyse Q4 expenses"})

# Runtime monitoring (SGP MGF)
handler.record_heartbeat()
handler.record_monitor(metric_name="requests_processed", value="50")

# Error handling (Integrity Oath)
handler.record_error(error_type="api_timeout", message="Retry succeeded on attempt 2")

# Human review (EU AI Act + SGP MGF)
handler.record_oversight(reviewer_id="finance-director", decision="approved", reason="Analysis verified")

handler.end_session()
```

## Documentation

- [agentapproved.ai](https://agentapproved.ai/) — homepage, getting started, pricing
- [EU AI Act compliance guide](https://agentapproved.ai/eu-ai-act.html)
- [Singapore MGF compliance guide](https://agentapproved.ai/singapore-mgf.html)
- [The Integrity Oath](https://agentapproved.ai/integrity.html)
- [PyPI package](https://pypi.org/project/agentapproved/)

## License

MIT — see [LICENSE](LICENSE)
