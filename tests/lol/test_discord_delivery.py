import asyncio
from types import SimpleNamespace

import pytest
from oracle_bets_core.evidence.settlement import SettlementResult
from oracle_bets_discord.bot import (
    _acquire_instance_lock,
    _is_owner,
    _proposal_message,
    _publication_intent,
    _publication_marker,
    _recover_message_id,
    _send_or_recover,
    _settlement_message,
    _unpublished_actionable_rows,
    _unpublished_open_position_rows,
    _unpublished_review_rows,
)
from oracle_bets_discord.delivery import (
    DiscordDeliveryMode,
    check_gateway_access,
    resolve_delivery_mode,
)
from oracle_bets_discord.formatting import DELIVERY_TARGET, split_message

EXPECTED_READ_CALLS = 3
EXPECTED_HISTORY_LIMIT = 100
RECOVERED_MESSAGE_ID = 2


def test_delivery_mode_is_explicit_and_gateway_prevents_webhook_duplication():
    environment = {
        "DISCORD_TOKEN": "token",
        "DISCORD_CHANNEL_ID": "123",
        "DISCORD_OWNER_USER_ID": "42",
        "DISCORD_WEBHOOK_URL": "https://discord.invalid/webhook",
    }

    assert resolve_delivery_mode("gateway", environment) is DiscordDeliveryMode.GATEWAY
    with pytest.raises(ValueError, match="gateway or off"):
        resolve_delivery_mode("webhook", environment)
    assert resolve_delivery_mode("off", environment) is DiscordDeliveryMode.OFF
    assert resolve_delivery_mode(None, environment) is DiscordDeliveryMode.GATEWAY


def test_delivery_mode_rejects_invalid_value():
    with pytest.raises(ValueError, match="gateway or off"):
        resolve_delivery_mode("both", {})


def test_explicit_empty_environment_does_not_fall_back_to_process_environment(
    monkeypatch,
):
    monkeypatch.setenv("DISCORD_DELIVERY_MODE", "gateway")

    assert resolve_delivery_mode(None, {}) is DiscordDeliveryMode.OFF


def test_message_splitter_preserves_content_and_limits_parts():
    content = "A" * 1800 + "\n\n" + "B" * 1800 + "\n\nfinal"
    parts = split_message(content)

    assert all(len(part) <= DELIVERY_TARGET for part in parts)
    assert "".join(parts) == content


def test_proposal_message_is_bounded_and_contains_only_review_facts():
    message = _proposal_message(
        {
            "proposal_id": "proposal-1",
            "run_id": "daily-1",
            "league": "LCK",
            "team_a": "T1",
            "team_b": "Gen.G",
            "target": "series_winner",
            "selection_id": "series_winner:::T1",
            "probability_point": "0.61",
            "stake_units": "1",
            "payload": {"odds": 2.0, "conservative_edge": 0.12},
        }
    )

    assert len(message) <= DELIVERY_TARGET
    assert "Paper proposal" in message
    assert "wallet" not in message.casefold()


def test_open_settlement_message_is_bounded_and_has_owner_choices():
    message = _settlement_message(
        {
            "position_id": "position-1",
            "league": "LCK",
            "team_a": "T1",
            "team_b": "Gen.G",
            "selection_id": "T1",
            "decimal_odds": "2.0",
            "stake_units": "1",
            "state": "open",
        }
    )

    assert len(message) <= DELIVERY_TARGET
    assert all(result.value.title() in message for result in SettlementResult)
    assert "LLM" not in message


def test_discord_controls_reject_non_owner_identity():
    assert _is_owner(42, 42)
    assert not _is_owner(41, 42)


def test_discord_publication_selects_only_unpublished_live_controls():
    pending = [
        {"proposal_id": "p1", "gate_state": "paper_actionable"},
        {"proposal_id": "p2", "gate_state": "blocked"},
        {"proposal_id": "p3", "gate_state": "research_only"},
    ]
    open_positions = [{"position_id": "open-1"}, {"position_id": "open-2"}]

    assert _unpublished_actionable_rows(pending, {"p1": 10}) == [pending[2]]
    assert _unpublished_open_position_rows(open_positions, {"open-2": 20}) == [
        open_positions[0]
    ]


def test_manual_review_publication_filters_completed_requests():
    requests = [
        {"review_id": "r1", "run_id": "run-1"},
        {"review_id": "r2", "run_id": "run-2"},
    ]

    assert _unpublished_review_rows(requests, {"r1": 10}) == [requests[1]]


def test_publication_intent_is_durable_before_send_and_marker_is_stable():
    class Store:
        def __init__(self):
            self.rows = []

        def append(self, table, row):
            self.rows.append((table, row))

        def get(self, table, record_id):
            del table, record_id

    store = Store()
    row = {"proposal_id": "proposal-1", "run_id": "daily-1"}

    intent = _publication_intent(store, row, kind="proposal")

    assert store.rows[0][1]["event_type"] == "discord_proposal_publish_intent"
    assert store.rows[0][1]["status"] == "pending"
    assert intent["marker"] == _publication_marker("proposal", "proposal-1")
    assert intent["marker"] in _proposal_message(row, marker=intent["marker"])

    long_row = row | {"league": "L" * 3000}
    bounded = _proposal_message(long_row, marker=intent["marker"])
    assert len(bounded) <= DELIVERY_TARGET
    assert bounded.endswith(intent["marker"])


def test_gateway_process_lock_rejects_second_instance(tmp_path):
    first = _acquire_instance_lock(tmp_path / "discord.lock")
    try:
        with pytest.raises(RuntimeError, match="already running"):
            _acquire_instance_lock(tmp_path / "discord.lock")
    finally:
        first.close()


def test_gateway_recovers_a_marked_message_without_resending():
    class Channel:
        async def history(self, **kwargs):
            assert kwargs["limit"] == EXPECTED_HISTORY_LIMIT
            for row in (
                SimpleNamespace(
                    id=1, content="unrelated", author=SimpleNamespace(id=7)
                ),
                SimpleNamespace(
                    id=9,
                    content="copied [oracle-ref:proposal:abc]",
                    author=SimpleNamespace(id=8),
                ),
                SimpleNamespace(
                    id=2,
                    content="card [oracle-ref:proposal:abc]",
                    author=SimpleNamespace(id=7),
                ),
            ):
                yield row

    recovered = asyncio.run(
        _recover_message_id(
            Channel(),
            marker="[oracle-ref:proposal:abc]",
            after="2026-08-10T10:00:00+00:00",
            author_id=7,
        )
    )

    assert recovered == RECOVERED_MESSAGE_ID


def test_gateway_send_then_crash_recovers_without_duplicate():
    class Store:
        def __init__(self):
            self.events = {}

        def get(self, _table, record_id):
            return self.events.get(record_id)

        def append(self, _table, row):
            self.events[row["id"]] = row

    class Channel:
        def __init__(self):
            self.messages = []

        async def history(self, **_kwargs):
            for message in self.messages:
                yield message

    store = Store()
    channel = Channel()
    row = {"proposal_id": "proposal-1", "run_id": "daily-1"}
    sends = 0

    async def scenario():
        nonlocal sends
        first_intent = _publication_intent(store, row, kind="proposal")

        async def send():
            nonlocal sends
            sends += 1
            message = SimpleNamespace(
                id=42,
                content=_proposal_message(row, marker=first_intent["marker"]),
                author=SimpleNamespace(id=7),
            )
            channel.messages.append(message)
            return message

        first = await _send_or_recover(
            channel,
            marker=first_intent["marker"],
            after=first_intent["intent_at"],
            author_id=7,
            send=send,
        )
        restarted_intent = _publication_intent(store, row, kind="proposal")
        second = await _send_or_recover(
            channel,
            marker=restarted_intent["marker"],
            after=restarted_intent["intent_at"],
            author_id=7,
            send=send,
        )
        return first, second

    first, second = asyncio.run(scenario())

    assert first == (42, True)
    assert second == (42, False)
    assert sends == 1


def test_live_doctor_uses_read_only_discord_requests():
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Session:
        def __init__(self):
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            if url.endswith("/users/@me"):
                return Response({"username": "Oracle"})
            if url.endswith("/messages"):
                return Response([])
            return Response({"name": "paper-bets"})

    session = Session()
    result = check_gateway_access("token", "123", session=session)

    assert result == {
        "bot": "Oracle",
        "channel": "paper-bets",
        "history_readable": "yes",
    }
    assert len(session.calls) == EXPECTED_READ_CALLS
