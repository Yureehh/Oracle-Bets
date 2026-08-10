"""Optional, bounded, advisory OpenAI review of deterministic proposals."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

from oracle_bets_core.operations.paper import ActionState

PROMPT_VERSION = "paper-review-v1"
MAX_INPUT_CHARS = 12_000
MAX_RATIONALE_CHARS = 300
MAX_REASON_CODE_CHARS = 60
MAX_OUTPUT_TOKENS = 1_000


@dataclass(frozen=True)
class AdvisoryReview:
    available: bool
    recommendations: tuple[dict[str, str], ...] = ()
    model_id: str | None = None
    response_id: str | None = None
    prompt_version: str = PROMPT_VERSION
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def review_proposals(
    proposals: list[dict[str, Any]],
    *,
    enabled: bool,
    model: str = "gpt-5.6-luna",
    client: Any | None = None,
) -> AdvisoryReview:
    """Make at most one no-tools call; never modify deterministic decisions."""
    eligible = [
        row for row in proposals if row.get("state") == ActionState.PAPER_ACTIONABLE
    ]
    if not enabled or not eligible:
        return AdvisoryReview(
            available=False, error="disabled_or_no_eligible_proposals"
        )
    compact_rows = [_compact_proposal(row) for row in eligible]
    compact = json.dumps(compact_rows, separators=(",", ":"), default=str)[
        :MAX_INPUT_CHARS
    ]
    started = time.monotonic()
    try:
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=20.0)
        response = client.responses.create(
            model=model,
            reasoning={"effort": "low"},
            max_output_tokens=MAX_OUTPUT_TOKENS,
            input=(
                "Review these already-gated paper proposals. Return advisory accept/refuse "
                "recommendations only. Do not change probabilities or stakes.\n"
                + compact
            ),
            text={
                "verbosity": "low",
                "format": {
                    "type": "json_schema",
                    "name": "paper_review",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "recommendations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "proposal_id": {"type": "string"},
                                        "recommendation": {
                                            "enum": ["accept", "refuse"]
                                        },
                                        "reason_code": {"type": "string"},
                                        "rationale": {
                                            "type": "string",
                                            "maxLength": MAX_RATIONALE_CHARS,
                                        },
                                    },
                                    "required": [
                                        "proposal_id",
                                        "recommendation",
                                        "reason_code",
                                        "rationale",
                                    ],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["recommendations"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        parsed = json.loads(response.output_text)
        recommendations = _validated_recommendations(
            parsed.get("recommendations"),
            eligible_ids={str(row["proposal_id"]) for row in eligible},
        )
        usage = getattr(response, "usage", None)
        return AdvisoryReview(
            available=True,
            recommendations=recommendations,
            model_id=model,
            response_id=response.id,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            latency_ms=round((time.monotonic() - started) * 1000),
        )
    except Exception as error:  # advisory failure is intentionally non-blocking
        return AdvisoryReview(
            available=False,
            model_id=model,
            latency_ms=round((time.monotonic() - started) * 1000),
            error=type(error).__name__,
        )


def format_advisory_review(review: AdvisoryReview) -> str | None:
    """Render bounded Discord copy; advisory output never controls a gate."""
    if not review.available or not review.recommendations:
        return None
    lines = ["**Optional AI review — advisory only**"]
    lines.extend(
        (
            f"- `{item['proposal_id']}`: **{item['recommendation']}** "
            f"({item['reason_code']}) — {item['rationale']}"
        )
        for item in review.recommendations
    )
    return "\n".join(lines)


def _compact_proposal(row: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "proposal_id",
        "fixture_key",
        "league",
        "team_a",
        "team_b",
        "target",
        "game_number",
        "total_line",
        "selection",
        "probability",
        "probability_lower",
        "decimal_odds",
        "conservative_edge",
        "stake_units",
        "state",
        "reason",
        "warnings",
        "drivers",
    )
    return {key: row.get(key) for key in allowed if row.get(key) is not None}


def _validated_recommendations(
    value: Any,
    *,
    eligible_ids: set[str],
) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list):
        raise TypeError("AI review recommendations must be a list")
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        proposal_id = str(item.get("proposal_id") or "")
        recommendation = str(item.get("recommendation") or "")
        if (
            proposal_id not in eligible_ids
            or proposal_id in seen
            or recommendation not in {"accept", "refuse"}
        ):
            continue
        seen.add(proposal_id)
        output.append(
            {
                "proposal_id": _sanitize(proposal_id, 100),
                "recommendation": recommendation,
                "reason_code": _sanitize(
                    str(item.get("reason_code") or "unspecified"),
                    MAX_REASON_CODE_CHARS,
                ),
                "rationale": _sanitize(
                    str(item.get("rationale") or "No rationale supplied."),
                    MAX_RATIONALE_CHARS,
                ),
            }
        )
    return tuple(output)


def _sanitize(value: str, limit: int) -> str:
    return " ".join(value.replace("@", "@\u200b").split())[:limit]
