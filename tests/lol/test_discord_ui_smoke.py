from types import SimpleNamespace

import pytest
from oracle_bets_core.evidence import EvidenceStore

HUB_BUTTON_COUNT = 7


def test_owner_console_components_construct_with_discord_py(tmp_path):
    discord = pytest.importorskip("discord")
    from oracle_bets_discord.ui.bets import build_bet_views
    from oracle_bets_discord.ui.common import build_common_views
    from oracle_bets_discord.ui.hub import build_hub_view
    from oracle_bets_discord.ui.performance import build_performance_view
    from oracle_bets_discord.ui.review import build_review_views

    store = EvidenceStore(tmp_path / "evidence.db")
    store.initialize_schema()
    logger = SimpleNamespace(exception=lambda *_args, **_kwargs: None)

    class Registry:
        @staticmethod
        def champion_id():
            return None

        @staticmethod
        def is_actionable(model_id):
            return model_id is not None

    registry = Registry()
    OwnerView, PageView = build_common_views(discord, 1)
    bets = build_bet_views(
        discord,
        store=store,
        owner_id=1,
        OwnerView=OwnerView,
        PageView=PageView,
    )
    reviews = build_review_views(
        discord,
        store=store,
        owner_id=1,
        logger=logger,
        OwnerView=OwnerView,
        BetOptionsView=bets.BetOptionsView,
    )
    PerformanceView = build_performance_view(
        discord,
        store=store,
        logger=logger,
        OwnerView=OwnerView,
    )
    OracleHub = build_hub_view(
        discord,
        store=store,
        registry=registry,
        logger=logger,
        OwnerView=OwnerView,
        PageView=PageView,
        ReviewLinksModal=reviews.ReviewLinksModal,
        BetOptionsView=bets.BetOptionsView,
        OpenBetsView=bets.OpenBetsView,
        PerformanceView=PerformanceView,
        RecentReviewsView=reviews.RecentReviewsView,
    )

    assert len(OracleHub().children) == HUB_BUTTON_COUNT
    assert PerformanceView().children
    assert reviews.RecentReviewsView().children
    assert (
        reviews.ReviewLinksModal(
            default_links="https://polymarket.com/test",
            review_key="review-key",
        ).links.default
        == "https://polymarket.com/test"
    )
