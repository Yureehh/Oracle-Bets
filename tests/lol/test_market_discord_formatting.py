DISCORD_MESSAGE_LIMIT = 2000


def test_oversized_single_line_is_truncated_in_schedule_chunks():
    from oracle_bets_discord.predictions.lol import _split_schedule_block

    giant_line = "x" * 3000
    block = "header\n" + giant_line + "\nfooter"

    chunks = _split_schedule_block(block, DISCORD_MESSAGE_LIMIT)

    assert all(len(chunk) < DISCORD_MESSAGE_LIMIT for chunk in chunks)
