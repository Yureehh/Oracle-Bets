"""
Team model: fast, safe access to team & player snapshots.

- Pins paired historical feature tables at the actual decision cutoff.
- Case-insensitive matching with .casefold() (better than .lower() for i18n).
- Partial roster updates allowed; missing roles autocompleted from last roster.
- Strong validation + clear errors, but minimal noise in logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

from oracle_bets_core.league_taxonomy import get_league_taxonomy
from oracle_bets_core.logger import logger
from oracle_bets_core.paths import PROCESSED_DIR, RATING_LEAGUE_ELO
from oracle_bets_core.pd import pd

from lol_bets.data_generation.feature_engineering.features_generator import (
    BREAK_THRESHOLD_DAYS,
)
from lol_bets.data_generation.feature_engineering.ratings_features.common import (
    is_cross_league_competition,
)
from lol_bets.data_generation.feature_engineering.ratings_features.elo import (
    apply_inactivity_decay as apply_elo_inactivity_decay,
)
from lol_bets.data_generation.feature_engineering.ratings_features.elo import (
    config as rating_config,
)
from lol_bets.data_generation.feature_engineering.ratings_features.elo import (
    handle_league_swap as swap_elo_league,
)
from lol_bets.data_generation.feature_engineering.ratings_features.glicko import (
    DEFAULT_MU,
    DEFAULT_PHI,
    DEFAULT_SIGMA,
    Rating,
)
from lol_bets.data_generation.feature_engineering.ratings_features.glicko import (
    apply_inactivity_decay as apply_glicko_inactivity_decay,
)
from lol_bets.data_generation.feature_engineering.ratings_features.glicko import (
    handle_league_swap as swap_glicko_league,
)
from lol_bets.data_generation.feature_engineering.ratings_features.plackett_luce import (
    DEFAULT_MU as PL_MU,
)
from lol_bets.data_generation.feature_engineering.ratings_features.plackett_luce import (
    DEFAULT_SIGMA as PL_SIGMA,
)
from lol_bets.data_generation.feature_engineering.ratings_features.plackett_luce import (
    handle_league_swap as swap_pl_league,
)
from lol_bets.data_generation.feature_engineering.ratings_features.plackett_luce import (
    initialize_pl_model,
)
from lol_bets.data_generation.feature_engineering.ratings_features.trueskill import (
    DEFAULT_MU as TS_MU,
)
from lol_bets.data_generation.feature_engineering.ratings_features.trueskill import (
    DEFAULT_SIGMA as TS_SIGMA,
)
from lol_bets.data_generation.feature_engineering.ratings_features.trueskill import (
    create_ts_rating,
)
from lol_bets.data_generation.feature_engineering.ratings_features.trueskill import (
    handle_league_swap as swap_ts_league,
)
from lol_bets.inference.roster import EXPECTED_ROLES
from lol_bets.inference.snapshots import FeatureSnapshot, load_feature_snapshot
from lol_bets.inference.team_resolver import TeamResolutionError, resolve_team_name

if TYPE_CHECKING:
    from collections.abc import Iterable


def _require_columns(df: pd.DataFrame, required: Iterable[str], where: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        msg = f"Missing columns in {where}: {missing}"
        raise ValueError(msg)


# ── Team ─────────────────────────────────────────────────────────────────── #

_EXPECTED_POS = EXPECTED_ROLES
_VALID_SIDES = {"blue": "Blue", "red": "Red"}
_DEFAULT_GLICKO_PHI = 350.0
_DEFAULT_SKILL_SIGMA = 8.333


class InsufficientRosterHistoryError(ValueError):
    """Raised when a team lacks enough player history for safe inference."""


def _normalize_side(side: str) -> str:
    normalized = _VALID_SIDES.get(side.strip().casefold())
    if normalized is None:
        msg = "Side must be either 'Blue' or 'Red'."
        raise ValueError(msg)
    return normalized


def _days_ago(value, as_of: pd.Timestamp) -> float:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return float("nan")
    if getattr(timestamp, "tzinfo", None) is not None:
        timestamp = timestamp.tz_convert(None)
    return float(max(0, (as_of - timestamp).days))


def _rating_uncertainty(row: pd.Series) -> float:
    values = []
    for column, baseline in (
        ("glicko2_phi", _DEFAULT_GLICKO_PHI),
        ("pl_sigma", _DEFAULT_SKILL_SIGMA),
        ("trueskill_sigma", _DEFAULT_SKILL_SIGMA),
    ):
        value = pd.to_numeric(row.get(column), errors="coerce")
        if pd.notna(value):
            values.append(float(value) / baseline)
    return float(sum(values) / len(values)) if values else float("nan")


@lru_cache(maxsize=2)
def _league_elo_values(mtime_ns: int, size: int) -> dict[str, float]:
    _ = mtime_ns, size
    frame = pd.read_parquet(RATING_LEAGUE_ELO, columns=["league", "elo"])
    return dict(zip(frame["league"].astype(str), frame["elo"], strict=True))


def _latest_rating_leagues(frame: pd.DataFrame, identity: str) -> dict[str, str]:
    if not {identity, "league", "date"}.issubset(frame.columns):
        return {}
    ordinary = frame.loc[
        frame["league"].notna()
        & ~frame["league"].astype(str).map(is_cross_league_competition)
    ]
    latest = ordinary.sort_values("date").drop_duplicates(identity, keep="last")
    return dict(
        zip(latest[identity].astype(str), latest["league"].astype(str), strict=True)
    )


def _reset_ratings_toward_baseline(row: pd.Series, retention: float) -> None:
    for column, baseline in (
        ("elo", float(rating_config["elo"]["initial"])),
        ("glicko2_mu", DEFAULT_MU),
        ("pl_mu", PL_MU),
        ("pl_sigma", PL_SIGMA),
        ("trueskill_mu", TS_MU),
        ("trueskill_sigma", TS_SIGMA),
    ):
        if pd.notna(row.get(column)):
            row[column] = baseline + (float(row[column]) - baseline) * retention


def _advance_inactive_ratings(
    row: pd.Series,
    as_of: pd.Timestamp,
    fixture_league: str | None = None,
    prior_league: str | None = None,
    prior_position: str | None = None,
) -> pd.Series:
    """Apply training season, league, inactivity, and role steps in order."""
    last_active = pd.to_datetime(row.get("date"), errors="coerce", utc=True)
    if pd.isna(last_active):
        return row
    last_active = last_active.tz_convert(None)
    frozen_league = row.get("_rating_league")
    if pd.notna(frozen_league):
        fixture_league = str(frozen_league)
    seasons = max(0, as_of.year - last_active.year)
    retention = float(rating_config["shared"]["decay_factor"]) ** seasons
    if seasons:
        _reset_ratings_toward_baseline(row, retention)
    if (
        fixture_league
        and prior_league
        and fixture_league != prior_league
        and not is_cross_league_competition(fixture_league)
    ):
        stat = RATING_LEAGUE_ELO.stat()
        league_elo = _league_elo_values(stat.st_mtime_ns, stat.st_size)
        transfer = float(rating_config["shared"]["transfer_factor"])
        identity = str(row.get("playerid") or row.get("teamid"))
        if pd.notna(row.get("elo")):
            elo_swap_state: dict[int | str, dict[str, Any]] = {
                identity: {"elo": float(row["elo"]), "league": prior_league}
            }
            swap_elo_league(
                identity,
                fixture_league,
                elo_swap_state,
                league_elo,
                float(rating_config["elo"]["initial"]),
                transfer,
            )
            row["elo"] = elo_swap_state[identity]["elo"]
        if pd.notna(row.get("glicko2_mu")) and pd.notna(row.get("glicko2_phi")):
            glicko_swap_state: dict[int | str, dict[str, Any]] = {
                identity: {
                    "rating": Rating(
                        float(row["glicko2_mu"]),
                        float(row["glicko2_phi"]),
                        DEFAULT_SIGMA,
                    ),
                    "league": prior_league,
                }
            }
            swap_glicko_league(
                identity,
                fixture_league,
                glicko_swap_state,
                league_elo,
                DEFAULT_MU,
                transfer,
            )
            row["glicko2_mu"] = glicko_swap_state[identity]["rating"].mu
        if pd.notna(row.get("pl_mu")) and pd.notna(row.get("pl_sigma")):
            pl_state: dict[int | str, dict[str, Any]] = {
                identity: {
                    "rating": initialize_pl_model(PL_MU, PL_SIGMA).rating(
                        float(row["pl_mu"]), float(row["pl_sigma"])
                    ),
                    "league": prior_league,
                }
            }
            swap_pl_league(
                identity,
                fixture_league,
                pl_state,
                league_elo,
                PL_MU,
                PL_SIGMA,
                transfer,
            )
            row["pl_mu"] = pl_state[identity]["rating"].mu
            row["pl_sigma"] = pl_state[identity]["rating"].sigma
        if pd.notna(row.get("trueskill_mu")) and pd.notna(row.get("trueskill_sigma")):
            ts_state: dict[int | str, dict[str, Any]] = {
                identity: {
                    "rating": create_ts_rating(
                        float(row["trueskill_mu"]), float(row["trueskill_sigma"])
                    ),
                    "league": prior_league,
                }
            }
            swap_ts_league(
                identity,
                fixture_league,
                ts_state,
                league_elo,
                TS_MU,
                TS_SIGMA,
                transfer,
            )
            row["trueskill_mu"] = ts_state[identity]["rating"].mu
            row["trueskill_sigma"] = ts_state[identity]["rating"].sigma
    if pd.notna(row.get("elo")):
        elo_state = {"elo": float(row["elo"]), "last_active": last_active}
        apply_elo_inactivity_decay(
            elo_state, as_of, float(rating_config["elo"]["initial"])
        )
        row["elo"] = elo_state["elo"]
    if pd.notna(row.get("glicko2_mu")) and pd.notna(row.get("glicko2_phi")):
        glicko_state = {
            "rating": Rating(
                mu=float(row["glicko2_mu"]),
                phi=float(row["glicko2_phi"]),
                sigma=DEFAULT_SIGMA,
            ),
            "last_active": last_active,
        }
        apply_glicko_inactivity_decay(glicko_state, as_of, DEFAULT_MU, DEFAULT_PHI)
        row["glicko2_mu"] = glicko_state["rating"].mu
        row["glicko2_phi"] = glicko_state["rating"].phi
    if prior_position and prior_position != row.get(
        "_rating_position", row.get("position")
    ):
        _reset_ratings_toward_baseline(
            row, 1.0 - float(rating_config["shared"]["position_reset_factor"])
        )
    return row


class TeamStateUnavailableError(ValueError):
    """The team has no completed historical state before this fixture."""


@dataclass
class Team:
    name: str
    side: str | None = None
    league: str | None = None
    first_pick: bool | None = None
    as_of_date: object | None = None
    decision_at: object | None = None
    feature_snapshot: FeatureSnapshot | None = None
    roster: dict[str, str | None] = field(
        default_factory=lambda: dict.fromkeys(_EXPECTED_POS)
    )

    # populated at init
    team_stats: pd.Series = field(init=False)
    player_stats: pd.DataFrame = field(init=False)
    established_roster: dict[str, str] = field(default_factory=dict, init=False)

    # cached tables
    _team_df: pd.DataFrame = field(init=False, repr=False)
    _player_df: pd.DataFrame = field(init=False, repr=False)
    _as_of: pd.Timestamp = field(init=False, repr=False)
    _team_activity_date: pd.Timestamp = field(init=False, repr=False)
    _established_roster_ids: set[str] = field(
        init=False, default_factory=set, repr=False
    )
    _team_rating_as_of: pd.Timestamp = field(init=False, repr=False)
    _player_rating_as_of: dict[str, pd.Timestamp] = field(init=False, repr=False)
    _player_activity_dates: dict[str, pd.Timestamp] = field(init=False, repr=False)
    _team_rating_league: str | None = field(init=False, repr=False)
    _player_rating_leagues: dict[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._load_snapshot()
        _require_columns(self._team_df, ["teamname"], "FLATTENED_TEAMS")
        _require_columns(
            self._player_df,
            ["teamname", "playername", "position", "date"],
            "FLATTENED_PLAYERS",
        )

        # set team_stats
        self.team_stats = self._lookup_team_row(self.name).copy()
        self.team_stats = _advance_inactive_ratings(
            self.team_stats,
            self._team_rating_as_of,
            self.league,
            self._team_rating_league,
        ).drop(labels=["_rating_league"])
        if self.league:
            self.team_stats["league"] = self.league
            self.team_stats["strength_pool"] = get_league_taxonomy(self.league)[
                "strength_pool"
            ]
        # Canonicalize the name so roster/player lookups use the Oracle's
        # Elixir teamname even when the caller passed an external alias.
        canonical_name = self.team_stats.get("teamname")
        if isinstance(canonical_name, str) and canonical_name:
            self.name = canonical_name
        if self.side:
            self.team_stats["side"] = _normalize_side(self.side)
            self.side = self.team_stats["side"]
        if self.first_pick is not None:
            self.team_stats["first_pick"] = int(bool(self.first_pick))

        # Complete roster from the latest known lineup, but retain how much of
        # the future lineup was supplied explicitly as an uncertainty signal.
        supplied_roles = {
            role
            for role, player in (self.roster or {}).items()
            if role in _EXPECTED_POS and isinstance(player, str) and player.strip()
        }
        try:
            last_roster = self._get_last_roster(self.name)
        except (ValueError, KeyError) as e:
            logger.debug("No last-roster fallback available: %s", e)
            last_roster = {}
        self.established_roster = dict(last_roster)
        filled = {**dict.fromkeys(_EXPECTED_POS), **(self.roster or {})}
        if any(v is None for v in filled.values()):
            for role in _EXPECTED_POS:
                if filled[role] is None and role in last_roster:
                    filled[role] = last_roster[role]

        missing_roles = [
            role
            for role, player in filled.items()
            if not isinstance(player, str) or not player.strip()
        ]
        if missing_roles:
            raise InsufficientRosterHistoryError(
                f"Insufficient historical roster coverage for '{self.name}': "
                f"missing {', '.join(missing_roles)}."
            )
        self.roster = filled
        self._validate_roster(strict=True)
        self.player_stats = self._lookup_players(self.roster)
        self.player_stats = self.player_stats.apply(
            lambda row: _advance_inactive_ratings(
                row,
                self._player_rating_as_of.get(
                    str(row.get("playerid", row.get("playername"))).casefold(),
                    self._as_of,
                ),
                self.league,
                self._player_rating_leagues.get(str(row.get("playerid"))),
                row.get("_prior_position"),
            ),
            axis=1,
        ).drop(columns=["_prior_position", "_rating_position", "_rating_league"])
        if self.league:
            self.player_stats["league"] = self.league
        current_members = {str(name).casefold() for name in self.roster.values()}
        prior_members = {str(name).casefold() for name in last_roster.values()}
        if self._established_roster_ids and "playerid" in self.player_stats:
            current_members = set(self.player_stats["playerid"].astype(str))
            prior_members = self._established_roster_ids
        continuity = len(current_members & prior_members) / len(_EXPECTED_POS)
        known_fraction = len(supplied_roles) / len(_EXPECTED_POS)
        self.team_stats["roster_continuity"] = continuity
        self.team_stats["roster_uncertainty"] = max(
            1.0 - continuity, 1.0 - known_fraction
        )
        self.team_stats["days_since_last_game"] = _days_ago(
            self._team_activity_date, self._as_of
        )
        self.team_stats["is_after_break"] = int(
            self.team_stats["days_since_last_game"] > BREAK_THRESHOLD_DAYS
        )
        self.team_stats["rating_uncertainty"] = _rating_uncertainty(self.team_stats)
        self.player_stats["days_since_last_game"] = self.player_stats.apply(
            lambda row: _days_ago(
                self._player_activity_dates[
                    str(
                        row.get("playerid")
                        if pd.notna(row.get("playerid"))
                        else row["playername"]
                    ).casefold()
                ],
                self._as_of,
            ),
            axis=1,
        )
        self.player_stats["rating_uncertainty"] = self.player_stats.apply(
            _rating_uncertainty, axis=1
        )

    def _load_snapshot(self) -> None:
        decision = pd.to_datetime(cast("Any", self.decision_at), utc=True)
        if pd.isna(decision):
            decision = pd.Timestamp.now(tz="UTC")
        if self.feature_snapshot is None:
            self.feature_snapshot = load_feature_snapshot(
                PROCESSED_DIR / "serving", decision_at=decision.to_pydatetime()
            )
        if (
            pd.to_datetime(self.feature_snapshot.manifest["observed_at"], utc=True)
            > decision
        ):
            raise ValueError("Feature snapshot not available at the decision time")
        self._team_df = self.feature_snapshot.read("teams")
        self._player_df = self.feature_snapshot.read("players")
        self._decision_at = decision
        self._as_of = pd.to_datetime(
            cast("Any", self.as_of_date), errors="coerce", utc=True
        )
        if pd.isna(self._as_of):
            self._as_of = decision
        if getattr(self._as_of, "tzinfo", None) is not None:
            self._as_of = self._as_of.tz_convert(None)

    # ── lookups ──────────────────────────────────────────────────────────── #

    def _at_or_before_as_of(
        self, df: pd.DataFrame, *, frozen_state: bool = False
    ) -> pd.DataFrame:
        as_of = getattr(self, "_as_of", None)
        if "date" not in df or as_of is None:
            return df
        column = "state_available_at" if "state_available_at" in df else "date"
        dates = pd.to_datetime(df[column], errors="coerce", utc=True)
        cutoff = pd.to_datetime(as_of, utc=True)
        decision = getattr(self, "_decision_at", None)
        if decision is not None:
            cutoff = min(cutoff, decision)
        if frozen_state and column == "state_available_at":
            day_start = pd.to_datetime(as_of, utc=True).normalize()
            starts = pd.to_datetime(df["date"], errors="coerce", utc=True)
            return df.loc[(starts < day_start) & (dates <= cutoff)]
        return df.loc[dates <= cutoff]

    def _frozen_rating_as_of(self, activity_rows: pd.DataFrame) -> pd.Timestamp:
        as_of = getattr(self, "_as_of", None)
        if as_of is None:
            return activity_rows["date"].max()
        starts = pd.to_datetime(activity_rows["date"], errors="coerce", utc=True)
        same_day = starts.loc[
            starts.dt.normalize().eq(pd.to_datetime(as_of, utc=True).normalize())
        ]
        # Training freezes the day's ratings at its first game, even when a
        # later game crosses another whole day of inactivity.
        return same_day.min().tz_convert(None) if not same_day.empty else as_of

    def _lookup_team_row(self, team_name: str) -> pd.Series:
        known_names = self._team_df["teamname"].dropna().astype(str).unique().tolist()
        resolved = resolve_team_name(team_name, known_names)
        if not resolved.ok:
            raise TeamResolutionError(resolved)
        resolved_name = resolved.resolved_name
        if resolved_name is None:
            raise TeamResolutionError(resolved)
        key = resolved_name.casefold()
        rows = self._team_df[self._team_df["teamname"].str.casefold() == key]
        activity_rows = self._at_or_before_as_of(rows)
        if activity_rows.empty:
            msg = f"Team '{team_name}' not found in teams table."
            raise TeamStateUnavailableError(msg)
        self._team_activity_date = activity_rows.sort_values("date").iloc[-1]["date"]
        self._team_rating_as_of = self._frozen_rating_as_of(activity_rows)
        rows = self._at_or_before_as_of(rows, frozen_state=True)
        if rows.empty:  # defensive: resolver guarantees membership
            msg = f"Team '{team_name}' not found in teams table."
            raise TeamStateUnavailableError(msg)
        rating_leagues = _latest_rating_leagues(rows, "teamid")
        if "date" in rows.columns:
            rows = rows.sort_values("date", ascending=False)
        selected = rows.iloc[0].copy()
        dates = pd.to_datetime(activity_rows["date"], utc=True)
        first = activity_rows.loc[
            dates.eq(pd.to_datetime(self._team_rating_as_of, utc=True))
        ]
        selected["_rating_league"] = (
            first.iloc[0].get("league") if not first.empty else None
        )
        self._team_rating_league = rating_leagues.get(str(selected.get("teamid")))
        return selected

    def _last_roster_from(self, df: pd.DataFrame, team_name: str) -> dict[str, str]:
        key = team_name.casefold()
        df = df[df["teamname"].str.casefold() == key]
        df = self._at_or_before_as_of(df)
        player_names = df["playername"].fillna("").astype(str).str.strip()
        df = df[player_names.ne("") & player_names.str.casefold().ne("nan")]
        if df.empty:
            msg = f"No player rows for team '{team_name}'."
            raise ValueError(msg)

        # latest per position
        latest = (
            df.sort_values(["position", "date"], ascending=[True, False])
            .groupby("position", observed=True)
            .first()
            .reset_index()
        )
        # ensure we have at least the expected roles
        missing = set(_EXPECTED_POS) - set(latest["position"].str.casefold())
        if missing:
            msg = f"Could not determine last roster for roles: {sorted(missing)}"
            raise ValueError(msg)

        self._established_roster_ids = (
            set(latest["playerid"].astype(str))
            if "playerid" in latest and latest["playerid"].notna().all()
            else set()
        )

        # build canonical mapping role -> playername
        out: dict[str, str] = {}
        for _, row in latest.iterrows():
            role = str(row["position"]).casefold()
            name = str(row["playername"])
            if role in _EXPECTED_POS:
                out[role] = name
        return out

    def _get_last_roster(self, team_name: str) -> dict[str, str]:
        return self._last_roster_from(self._player_df, team_name)

    def _lookup_players(self, roster: dict[str, str | None]) -> pd.DataFrame:
        # expect fully-populated roster already validated
        wanted = {role: (name or "") for role, name in roster.items()}
        wanted_lower = {role: name.casefold() for role, name in wanted.items()}
        requested_names = set(wanted_lower.values())
        team_key = self.name.casefold()

        df = self._player_df[
            self._player_df["playername"].str.casefold().isin(requested_names)
        ].copy()
        activity_rows = self._at_or_before_as_of(df)
        activity_key = "playerid" if "playerid" in activity_rows else "playername"
        self._player_activity_dates = (
            activity_rows.groupby(
                activity_rows[activity_key].astype(str).str.casefold()
            )["date"]
            .max()
            .to_dict()
        )
        self._player_rating_as_of = {
            str(identity).casefold(): self._frozen_rating_as_of(rows)
            for identity, rows in activity_rows.groupby(activity_key)
        }
        rating_positions: dict[str, str] = {}
        rating_leagues: dict[str, str] = {}
        as_of = getattr(self, "_as_of", None)
        if as_of is not None:
            dates = pd.to_datetime(activity_rows["date"], utc=True)
            day = pd.to_datetime(as_of, utc=True).normalize()
            first = (
                activity_rows.loc[dates.dt.normalize().eq(day)]
                .sort_values("date")
                .drop_duplicates(activity_key, keep="first")
            )
            rating_positions = dict(
                zip(
                    first[activity_key].astype(str).str.casefold(),
                    first["position"],
                    strict=True,
                )
            )
            if "league" in first:
                rating_leagues = dict(
                    zip(
                        first[activity_key].astype(str).str.casefold(),
                        first["league"],
                        strict=True,
                    )
                )
        df = self._at_or_before_as_of(df, frozen_state=True)
        self._player_rating_leagues = _latest_rating_leagues(df, "playerid")

        if df.empty:
            msg = f"No player stats found for roster of team '{self.name}'."
            raise InsufficientRosterHistoryError(msg)

        df["_player_key"] = df["playername"].str.casefold()
        df["_team_match"] = df["teamname"].str.casefold().eq(team_key)

        # Team history disambiguates identical names. State follows the resolved
        # player identity globally, including spells with a different team.
        team_rows = (
            df[df["_team_match"]]
            .sort_values(["_player_key", "date"], ascending=[True, False])
            .drop_duplicates(subset=["_player_key"], keep="first")
        )
        if "playerid" in df:
            identities = team_rows[["_player_key", "playerid"]]
            team_rows = (
                df.merge(
                    identities, on=["_player_key", "playerid"], validate="many_to_one"
                )
                .sort_values("date", ascending=False)
                .drop_duplicates("_player_key", keep="first")
            )
        found_team_rows = set(team_rows["_player_key"])
        fallback_rows = (
            df[~df["_player_key"].isin(found_team_rows)]
            .sort_values(["_player_key", "date"], ascending=[True, False])
            .drop_duplicates(subset=["_player_key"], keep="first")
        )
        df = pd.concat([team_rows, fallback_rows], ignore_index=True).drop(
            columns=["_player_key", "_team_match"]
        )

        # ensure we have one row per requested player
        have_lower = set(df["playername"].str.casefold())
        missing = [nm for nm in requested_names if nm not in have_lower]
        if missing:
            msg = f"Missing statistics for players: {sorted(set(missing))}"
            raise InsufficientRosterHistoryError(msg)

        # attach canonical roles (by matching case-insensitive names back to roster)
        name_to_role = {wanted_lower[r]: r for r in _EXPECTED_POS}
        df["role"] = df["playername"].str.casefold().map(name_to_role).fillna("unknown")
        if set(df["role"]) != set(_EXPECTED_POS) or df["role"].duplicated().any():
            msg = f"Roster for '{self.name}' does not resolve to exactly one player per role."
            raise ValueError(msg)
        df["_rating_position"] = (
            df[activity_key]
            .astype(str)
            .str.casefold()
            .map(rating_positions)
            .fillna(df["role"])
        )
        df["_rating_league"] = (
            df[activity_key].astype(str).str.casefold().map(rating_leagues)
        )
        df["_prior_position"] = df["position"]
        df["position"] = df["role"]

        # re-order helpful columns if present
        cols = ["role", "playername", "teamname", "position", "date"]
        ordered = [c for c in cols if c in df.columns] + [
            c for c in df.columns if c not in cols
        ]
        return df[ordered]

    def _validate_roster(self, *, strict: bool = True) -> None:
        roles = set(self.roster.keys())
        missing_roles = [r for r in _EXPECTED_POS if r not in roles]
        if missing_roles:
            msg = f"Roster missing required roles: {missing_roles}"
            raise ValueError(msg)

        values = list(self.roster.values())
        if strict and any(
            v is None or not isinstance(v, str) or not v.strip() for v in values
        ):
            msg = "Roster must provide a non-empty player name for every role."
            raise ValueError(msg)

        # warn on duplicates (same name in multiple roles)
        names_norm = [str(v).casefold() for v in values if isinstance(v, str)]
        if len(names_norm) != len(set(names_norm)):
            logger.warning(
                "Roster for '%s' contains duplicate player names.", self.name
            )
