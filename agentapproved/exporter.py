"""Evidence packet generator — the auditor-facing export.

Generates a self-contained evidence bundle:
  - evidence.json     Machine-readable: all events + metadata + integrity proof
  - report.html       Human-readable: article-by-article compliance report
  - integrity.json    Standalone chain + signature verification data

All three files are self-contained. An auditor can verify integrity
without access to the AgentApproved platform.
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .hasher import get_public_key_hex, verify_signature
from .mapper import ComplianceReport, assess_compliance
from .schema import EvidenceEvent
from .transport import verify_session_file


# ── Evidence Packet ─────────────────────────────────────────────


def generate_evidence_packet(
    events: list[EvidenceEvent],
    output_dir: str | Path,
    organisation: str = "Organisation Name",
    public_key_hex: str = "",
) -> Path:
    """Generate a complete evidence packet from a list of events.

    Creates output_dir/ with:
        evidence.json    — full event data + metadata
        report.html      — human-readable compliance report
        integrity.json   — chain verification proof

    Returns the output_dir Path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = assess_compliance(events)
    chain_valid, chain_count = _verify_chain(events, public_key_hex)

    # 1. Machine-readable evidence
    evidence_doc = _build_evidence_json(events, report, public_key_hex, organisation)
    (output_dir / "evidence.json").write_text(
        json.dumps(evidence_doc, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    # 2. Integrity proof
    integrity_doc = _build_integrity_json(events, chain_valid, public_key_hex)
    (output_dir / "integrity.json").write_text(
        json.dumps(integrity_doc, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    # 3. Human-readable HTML report
    html = _build_report_html(report, events, chain_valid, public_key_hex, organisation)
    (output_dir / "report.html").write_text(html, encoding="utf-8")

    return output_dir


# ── JSON Builders ───────────────────────────────────────────────


def _build_evidence_json(
    events: list[EvidenceEvent],
    report: ComplianceReport,
    public_key_hex: str,
    organisation: str,
) -> dict:
    period_start = events[0].timestamp if events else ""
    period_end = events[-1].timestamp if events else ""

    return {
        "format": "agentapproved-evidence-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "organisation": organisation,
        "period": {"start": period_start, "end": period_end},
        "public_key": public_key_hex,
        "compliance": report.to_dict(),
        "event_count": len(events),
        "events": [e.to_dict() for e in events],
    }


def _build_integrity_json(
    events: list[EvidenceEvent],
    chain_valid: bool,
    public_key_hex: str,
) -> dict:
    return {
        "format": "agentapproved-integrity-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chain_valid": chain_valid,
        "event_count": len(events),
        "chain_start": events[0].previous_hash if events else "GENESIS",
        "chain_end": events[-1].event_hash if events else "",
        "public_key": public_key_hex,
        "signatures_present": all(e.signature for e in events),
        "events_summary": [
            {
                "sequence": e.sequence_number,
                "event_id": e.event_id,
                "action_type": e.action_type,
                "event_hash": e.event_hash,
                "signature": e.signature[:16] + "..." if e.signature else "",
            }
            for e in events
        ],
    }


# ── Chain Verification ──────────────────────────────────────────


def _verify_chain(
    events: list[EvidenceEvent], public_key_hex: str
) -> tuple[bool, int]:
    """Verify hash chain + signatures in-memory."""
    from .hasher import compute_event_hash, public_key_from_hex

    pub_key = None
    if public_key_hex:
        try:
            pub_key = public_key_from_hex(public_key_hex)
        except Exception:
            pass

    for i, event in enumerate(events):
        expected_prev = "GENESIS" if i == 0 else events[i - 1].event_hash
        if event.previous_hash != expected_prev:
            return False, i
        computed = compute_event_hash(event.to_hashable_dict())
        if computed != event.event_hash:
            return False, i
        if pub_key and event.signature:
            if not verify_signature(event.event_hash, event.signature, pub_key):
                return False, i

    return True, len(events)


# ── HTML Report Builder ─────────────────────────────────────────


_STATUS_ICONS = {
    "satisfied": "&#10003;",      # ✓
    "partial": "&#9888;",         # ⚠
    "missing": "&#10007;",        # ✗
    "not_applicable": "&#8212;",  # —
}

_STATUS_COLORS = {
    "satisfied": "#16a34a",
    "partial": "#d97706",
    "missing": "#dc2626",
    "not_applicable": "#6b7280",
}

_STATUS_BG = {
    "satisfied": "#f0fdf4",
    "partial": "#fffbeb",
    "missing": "#fef2f2",
    "not_applicable": "#f9fafb",
}


def _build_report_html(
    report: ComplianceReport,
    events: list[EvidenceEvent],
    chain_valid: bool,
    public_key_hex: str,
    organisation: str,
) -> str:
    period_start = events[0].timestamp[:10] if events else "N/A"
    period_end = events[-1].timestamp[:10] if events else "N/A"
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Event type breakdown
    type_counts: dict[str, int] = {}
    for e in events:
        type_counts[e.action_type] = type_counts.get(e.action_type, 0) + 1
    type_rows = "\n".join(
        f"<tr><td>{t}</td><td style='text-align:right'>{c}</td></tr>"
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1])
    )

    # Requirement rows
    req_rows = ""
    for req in report.requirements:
        icon = _STATUS_ICONS[req.status]
        color = _STATUS_COLORS[req.status]
        bg = _STATUS_BG[req.status]
        remediation_html = ""
        if req.remediation:
            remediation_html = f"""
            <div style="margin-top:8px;padding:10px;background:#f8f9fa;
                        border-left:3px solid {color};font-size:13px;color:#374151">
                <strong>Remediation:</strong> {_esc(req.remediation)}
            </div>"""

        req_rows += f"""
        <div style="margin-bottom:16px;padding:16px;background:{bg};
                    border:1px solid #e5e7eb;border-radius:8px">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                    <span style="font-size:20px;color:{color};margin-right:8px">{icon}</span>
                    <strong>{_esc(req.article)}</strong> &mdash; {_esc(req.title)}
                </div>
                <span style="background:{color};color:white;padding:2px 10px;
                             border-radius:12px;font-size:12px;font-weight:600;
                             text-transform:uppercase">{req.status}</span>
            </div>
            <p style="margin:8px 0 0 30px;color:#4b5563;font-size:14px">
                {_esc(req.description)}
            </p>
            <p style="margin:4px 0 0 30px;color:#6b7280;font-size:13px">
                Evidence events: {req.evidence_count}
            </p>
            {remediation_html}
        </div>"""

    # Score color
    if report.overall_score >= 80:
        score_color = "#16a34a"
    elif report.overall_score >= 50:
        score_color = "#d97706"
    else:
        score_color = "#dc2626"

    chain_badge = (
        '<span style="background:#16a34a;color:white;padding:2px 10px;'
        'border-radius:12px;font-size:13px">VERIFIED</span>'
        if chain_valid
        else '<span style="background:#dc2626;color:white;padding:2px 10px;'
        'border-radius:12px;font-size:13px">BROKEN</span>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EU AI Act Compliance Evidence — AgentApproved</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         color:#111827; background:#ffffff; max-width:800px; margin:0 auto;
         padding:40px 24px; line-height:1.6; }}
  h1 {{ font-size:24px; margin-bottom:4px; }}
  h2 {{ font-size:18px; margin:32px 0 16px; padding-bottom:8px;
        border-bottom:2px solid #e5e7eb; }}
  table {{ width:100%; border-collapse:collapse; margin:12px 0; }}
  th, td {{ padding:8px 12px; text-align:left; border-bottom:1px solid #e5e7eb;
            font-size:14px; }}
  th {{ background:#f9fafb; font-weight:600; color:#374151; }}
  .meta {{ color:#6b7280; font-size:14px; }}
  @media print {{
    body {{ padding:20px; }}
    h2 {{ break-before:auto; }}
  }}
</style>
</head>
<body>

<div style="text-align:center;margin-bottom:32px">
  <div style="font-size:13px;text-transform:uppercase;letter-spacing:2px;
              color:#6b7280;margin-bottom:8px">AgentApproved</div>
  <h1>EU AI Act Article 12<br>Compliance Evidence Report</h1>
  <div class="meta" style="margin-top:12px">
    <div><strong>Organisation:</strong> {_esc(organisation)}</div>
    <div><strong>Period:</strong> {period_start} to {period_end}</div>
    <div><strong>Generated:</strong> {generated}</div>
  </div>
</div>

<div style="text-align:center;margin:24px 0 32px">
  <div style="display:inline-block;padding:20px 40px;border:3px solid {score_color};
              border-radius:16px">
    <div style="font-size:48px;font-weight:700;color:{score_color}">{report.overall_score}%</div>
    <div style="font-size:14px;color:#6b7280">Article 12 Compliance</div>
  </div>
</div>

<div style="display:flex;gap:12px;justify-content:center;margin-bottom:32px;flex-wrap:wrap">
  <div style="padding:8px 16px;background:#f0fdf4;border-radius:8px;text-align:center">
    <div style="font-size:20px;font-weight:700;color:#16a34a">{report.satisfied}</div>
    <div style="font-size:12px;color:#6b7280">Satisfied</div>
  </div>
  <div style="padding:8px 16px;background:#fffbeb;border-radius:8px;text-align:center">
    <div style="font-size:20px;font-weight:700;color:#d97706">{report.partial}</div>
    <div style="font-size:12px;color:#6b7280">Partial</div>
  </div>
  <div style="padding:8px 16px;background:#fef2f2;border-radius:8px;text-align:center">
    <div style="font-size:20px;font-weight:700;color:#dc2626">{report.missing}</div>
    <div style="font-size:12px;color:#6b7280">Missing</div>
  </div>
  <div style="padding:8px 16px;background:#f9fafb;border-radius:8px;text-align:center">
    <div style="font-size:20px;font-weight:700;color:#6b7280">{report.not_applicable}</div>
    <div style="font-size:12px;color:#6b7280">N/A</div>
  </div>
</div>

<h2>Article-by-Article Assessment</h2>
{req_rows}

<h2>Event Summary</h2>
<table>
  <tr><th>Metric</th><th style="text-align:right">Value</th></tr>
  <tr><td>Total events captured</td><td style="text-align:right">{len(events)}</td></tr>
  <tr><td>Sessions</td><td style="text-align:right">{len(set(e.session_id for e in events))}</td></tr>
  <tr><td>Unique agents</td>
      <td style="text-align:right">{len(set(e.actor_id for e in events if e.actor_type == 'agent'))}</td></tr>
  <tr><td>Human oversight events</td>
      <td style="text-align:right">{sum(1 for e in events if e.action_type == 'human_oversight')}</td></tr>
</table>

<h2>Event Type Breakdown</h2>
<table>
  <tr><th>Event Type</th><th style="text-align:right">Count</th></tr>
  {type_rows}
</table>

<h2>Integrity Verification</h2>
<table>
  <tr><td>Hash chain status</td><td style="text-align:right">{chain_badge}</td></tr>
  <tr><td>Events in chain</td><td style="text-align:right">{len(events)}</td></tr>
  <tr><td>Chain start</td>
      <td style="text-align:right;font-family:monospace;font-size:12px">
        {events[0].previous_hash if events else 'N/A'}</td></tr>
  <tr><td>Chain end</td>
      <td style="text-align:right;font-family:monospace;font-size:12px">
        {events[-1].event_hash[:32] + '...' if events else 'N/A'}</td></tr>
  <tr><td>Signatures present</td>
      <td style="text-align:right">{'Yes' if all(e.signature for e in events) else 'No'}</td></tr>
  <tr><td>Public key</td>
      <td style="text-align:right;font-family:monospace;font-size:12px">
        {public_key_hex[:32] + '...' if public_key_hex else 'Not available'}</td></tr>
</table>

<div style="margin-top:40px;padding:16px;background:#f9fafb;border-radius:8px;
            font-size:12px;color:#6b7280;text-align:center">
  This report was generated by <strong>AgentApproved</strong> (agentapproved.ai).
  Evidence integrity can be verified independently using the accompanying
  <code>evidence.json</code> and <code>integrity.json</code> files.
</div>

</body>
</html>"""


def _esc(text: str) -> str:
    """Basic HTML escaping."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
