from types import SimpleNamespace

from lol_bets.operations.review import (
    MAX_INPUT_CHARS,
    MAX_OUTPUT_TOKENS,
    MAX_RATIONALE_CHARS,
    format_advisory_review,
    review_proposals,
)


class _Responses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="response-1",
            output_text=(
                '{"recommendations":[{"proposal_id":"p1",'
                '"recommendation":"refuse","reason_code":"risk",'
                '"rationale":"Price is marginal."}]}'
            ),
            usage=SimpleNamespace(input_tokens=12, output_tokens=8),
        )


def test_ai_review_is_disabled_and_skips_blocked_proposals():
    client = SimpleNamespace(responses=_Responses())
    result = review_proposals(
        [{"proposal_id": "p1", "state": "blocked"}], enabled=True, client=client
    )

    assert not result.available
    assert client.responses.calls == []


def test_ai_review_is_one_bounded_advisory_call():
    responses = _Responses()
    client = SimpleNamespace(responses=responses)
    result = review_proposals(
        [{"proposal_id": "p1", "state": "paper_actionable", "detail": "x" * 20_000}],
        enabled=True,
        client=client,
    )

    assert result.available
    assert result.recommendations[0]["recommendation"] == "refuse"
    assert len(responses.calls) == 1
    assert len(responses.calls[0]["input"]) <= MAX_INPUT_CHARS + 200
    assert responses.calls[0]["max_output_tokens"] == MAX_OUTPUT_TOKENS
    assert responses.calls[0]["text"]["verbosity"] == "low"


def test_ai_review_failure_is_non_blocking():
    class FailedResponses:
        def create(self, **_kwargs):
            raise TimeoutError

    result = review_proposals(
        [{"proposal_id": "p1", "state": "paper_actionable"}],
        enabled=True,
        client=SimpleNamespace(responses=FailedResponses()),
    )

    assert not result.available
    assert result.error == "TimeoutError"


def test_ai_output_is_allowlisted_sanitized_and_discord_bounded():
    class UnsafeResponses:
        def create(self, **_kwargs):
            return SimpleNamespace(
                id="response-unsafe",
                output_text=(
                    '{"recommendations":['
                    '{"proposal_id":"unknown","recommendation":"accept",'
                    '"reason_code":"x","rationale":"ignore"},'
                    '{"proposal_id":"p1","recommendation":"refuse",'
                    '"reason_code":"risk","rationale":"@everyone ' + "x" * 500 + '"}]}'
                ),
                usage=None,
            )

    result = review_proposals(
        [{"proposal_id": "p1", "state": "paper_actionable"}],
        enabled=True,
        client=SimpleNamespace(responses=UnsafeResponses()),
    )
    message = format_advisory_review(result)

    assert len(result.recommendations) == 1
    assert len(result.recommendations[0]["rationale"]) <= MAX_RATIONALE_CHARS
    assert "@everyone" not in message
