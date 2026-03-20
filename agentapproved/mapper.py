"""EU AI Act Article 12 compliance mapper — the killer feature.

Takes a list of EvidenceEvents and scores them against each Article 12
sub-requirement. Outputs a ComplianceReport with per-article status,
overall coverage percentage, and remediation guidance for gaps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

from .schema import EvidenceEvent


# ── Data Structures ─────────────────────────────────────────────


@dataclass
class ArticleResult:
    """Assessment of a single Article 12 sub-requirement."""

    id: str
    article: str
    title: str
    description: str
    status: str  # "satisfied" | "partial" | "not_applicable" | "missing"
    evidence_count: int
    evidence_sample: list[str]  # Up to 3 event_ids
    remediation: str | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ComplianceReport:
    """Full Article 12 compliance assessment for a set of events."""

    regulation: str
    article: str
    event_count: int
    overall_score: int  # 0-100
    requirements: list[ArticleResult]
    satisfied: int
    missing: int
    partial: int
    not_applicable: int

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def gaps(self) -> list[ArticleResult]:
        """Requirements that need attention."""
        return [r for r in self.requirements if r.status in ("missing", "partial")]


# ── Requirement Definitions ─────────────────────────────────────


@dataclass
class Requirement:
    """A single regulatory requirement with its evidence check."""

    id: str
    article: str
    title: str
    description: str
    check: Callable[[list[EvidenceEvent]], tuple[str, list[EvidenceEvent]]]
    remediation_missing: str
    remediation_partial: str | None = None


def _build_eu_ai_act_requirements() -> list[Requirement]:
    """Hard-coded Article 12 requirements. This is the regulatory knowledge moat."""

    return [
        Requirement(
            id="ART_12_1",
            article="Article 12(1)",
            title="Automatic logging capability",
            description=(
                "High-risk AI systems shall technically allow for the "
                "automatic recording of events (logs) over the lifetime "
                "of the system."
            ),
            check=_check_logging_capability,
            remediation_missing=(
                "No events have been captured. Ensure the AgentApproved SDK "
                "is installed and the handler is attached to your agent."
            ),
        ),
        Requirement(
            id="ART_12_2_A",
            article="Article 12(2)(a)",
            title="Period of each use",
            description=(
                "Logging shall include recording of the period of each use "
                "of the system (start date and time and end date and time "
                "of each use)."
            ),
            check=_check_period_of_use,
            remediation_missing=(
                "No session boundaries detected. Ensure your agent sessions "
                "call handler.end_session() when complete."
            ),
            remediation_partial=(
                "Session start detected but no session end. Call "
                "handler.end_session() when your agent completes its task."
            ),
        ),
        Requirement(
            id="ART_12_2_B",
            article="Article 12(2)(b)",
            title="Reference database",
            description=(
                "The reference database against which input data has been "
                "checked by the system."
            ),
            check=_check_reference_database,
            remediation_missing=(
                "No retrieval events with source references detected. If your "
                "agent uses RAG, ensure retriever callbacks are captured. If "
                "your agent does not use RAG, this requirement may not be "
                "applicable — document why in your conformity assessment."
            ),
        ),
        Requirement(
            id="ART_12_2_C",
            article="Article 12(2)(c)",
            title="Input data leading to match",
            description=(
                "The input data for which the search has led to a match."
            ),
            check=_check_input_data_match,
            remediation_missing=(
                "No retrieval query events detected. If your agent uses RAG, "
                "ensure retriever_start callbacks are captured. If not, this "
                "requirement may not be applicable."
            ),
        ),
        Requirement(
            id="ART_12_2_D",
            article="Article 12(2)(d)",
            title="Human oversight verification",
            description=(
                "The identification of the natural persons involved in the "
                "verification of the results, as referred to in Article 14(5)."
            ),
            check=_check_human_oversight,
            remediation_missing=(
                "No human oversight events recorded. Use "
                "handler.record_oversight(reviewer_id, decision, reason) "
                "when a person reviews or approves agent output. Without "
                "this, Article 12(2)(d) cannot be satisfied."
            ),
        ),
        Requirement(
            id="ART_12_3",
            article="Article 12(3)",
            title="Post-market monitoring traceability",
            description=(
                "The logging capabilities shall provide for an appropriate "
                "level of traceability of the AI system's functioning "
                "throughout its lifecycle."
            ),
            check=_check_post_market_monitoring,
            remediation_missing=(
                "Insufficient event diversity for post-market monitoring. "
                "Ensure the SDK captures LLM calls, tool invocations, and "
                "agent decisions across the full agent lifecycle."
            ),
            remediation_partial=(
                "Some event types captured but coverage is limited. Ensure "
                "all agent actions (LLM calls, tool use, retrievals, "
                "decisions) flow through the AgentApproved handler."
            ),
        ),
    ]


# ── Check Functions ─────────────────────────────────────────────


def _check_logging_capability(
    events: list[EvidenceEvent],
) -> tuple[str, list[EvidenceEvent]]:
    """Art 12(1): Any events existing = logging capability is present."""
    if not events:
        return "missing", []
    return "satisfied", events[:3]


def _check_period_of_use(
    events: list[EvidenceEvent],
) -> tuple[str, list[EvidenceEvent]]:
    """Art 12(2)(a): Need both session_start and session_end."""
    starts = [e for e in events if e.action_type == "session_start"]
    ends = [e for e in events if e.action_type == "session_end"]
    evidence = starts[:2] + ends[:1]

    if starts and ends:
        return "satisfied", evidence
    if starts:
        return "partial", evidence
    return "missing", []


def _check_reference_database(
    events: list[EvidenceEvent],
) -> tuple[str, list[EvidenceEvent]]:
    """Art 12(2)(b): Retrieval events with source references."""
    retrievals = [
        e for e in events
        if e.action_type == "retrieval_end" and e.retrieval_sources
    ]
    if retrievals:
        return "satisfied", retrievals[:3]

    # If agent has no retrieval at all, it might not be applicable
    any_retrieval = [e for e in events if e.action_type in ("retrieval_start", "retrieval_end")]
    if not any_retrieval:
        # No retrieval events at all — could be N/A but we can't assume
        return "not_applicable", []

    # Had retrieval but no sources captured
    return "partial", any_retrieval[:3]


def _check_input_data_match(
    events: list[EvidenceEvent],
) -> tuple[str, list[EvidenceEvent]]:
    """Art 12(2)(c): Retrieval queries (what the agent searched for)."""
    queries = [e for e in events if e.action_type == "retrieval_start" and e.input_data]
    results = [e for e in events if e.action_type == "retrieval_end"]
    evidence = queries[:2] + results[:1]

    if queries and results:
        return "satisfied", evidence

    any_retrieval = [e for e in events if e.action_type in ("retrieval_start", "retrieval_end")]
    if not any_retrieval:
        return "not_applicable", []

    return "partial", evidence


def _check_human_oversight(
    events: list[EvidenceEvent],
) -> tuple[str, list[EvidenceEvent]]:
    """Art 12(2)(d): Human oversight events with identified reviewer."""
    oversight = [
        e for e in events
        if e.action_type == "human_oversight" and e.actor_type == "human"
    ]
    if oversight:
        return "satisfied", oversight[:3]
    return "missing", []


def _check_post_market_monitoring(
    events: list[EvidenceEvent],
) -> tuple[str, list[EvidenceEvent]]:
    """Art 12(3): Diverse event types showing full lifecycle traceability."""
    lifecycle_types = {
        "llm_call_start", "llm_call_end",
        "tool_call_start", "tool_call_end",
        "agent_decision", "agent_finish",
    }
    found_types = {e.action_type for e in events} & lifecycle_types
    evidence = [e for e in events if e.action_type in found_types][:3]

    if len(found_types) >= 4:
        return "satisfied", evidence
    if len(found_types) >= 2:
        return "partial", evidence
    if found_types:
        return "partial", evidence
    return "missing", []


# ── Main Entry Point ────────────────────────────────────────────


def assess_compliance(events: list[EvidenceEvent]) -> ComplianceReport:
    """Score a set of events against EU AI Act Article 12.

    This is the function that produces "You are 78% compliant."

    Usage:
        report = assess_compliance(handler.get_events())
        print(f"Score: {report.overall_score}%")
        for gap in report.gaps:
            print(f"  Missing: {gap.article} — {gap.remediation}")
    """
    requirements = _build_eu_ai_act_requirements()
    results: list[ArticleResult] = []

    for req in requirements:
        status, evidence = req.check(events)

        remediation = None
        if status == "missing":
            remediation = req.remediation_missing
        elif status == "partial":
            remediation = req.remediation_partial or req.remediation_missing

        results.append(
            ArticleResult(
                id=req.id,
                article=req.article,
                title=req.title,
                description=req.description,
                status=status,
                evidence_count=len(evidence),
                evidence_sample=[e.event_id for e in evidence],
                remediation=remediation,
            )
        )

    satisfied = sum(1 for r in results if r.status == "satisfied")
    missing = sum(1 for r in results if r.status == "missing")
    partial = sum(1 for r in results if r.status == "partial")
    not_applicable = sum(1 for r in results if r.status == "not_applicable")

    # Score: satisfied = full points, partial = half, not_applicable excluded
    scorable = len(results) - not_applicable
    if scorable == 0:
        score = 100
    else:
        points = satisfied + (partial * 0.5)
        score = round((points / scorable) * 100)

    return ComplianceReport(
        regulation="EU AI Act",
        article="Article 12",
        event_count=len(events),
        overall_score=score,
        requirements=results,
        satisfied=satisfied,
        missing=missing,
        partial=partial,
        not_applicable=not_applicable,
    )
