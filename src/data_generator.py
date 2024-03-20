"""
Data Generator.

This script is intended to represent the main function to update data once per day.
It takes the data that is daily stored in an S3 bucket from Oracle Elixir
and enriches it with additional data based on computed ratings.
The enriched data is then stored in the processed directory.


Please visit and support www.oracleselixir.com
Tim provides an invaluable service to the League community.
"""
# Housekeeping
import datetime as dt
import json
from dataclasses import dataclass, field
from os import getenv

import boto3
import pandas as pd
from dotenv import load_dotenv

from src.data_ingest.oracles_elixir import OraclesElixir
from src.feature_engineering.impute_early_game_metrics import EarlyGameStatsImputer
from src.performance_features.performance_metrics import PerformanceMetrics
from src.ratings_features.rating_models import Ratings
from utils.logger import logger
from utils.paths import INTERIM_DIR, INVALID_GAMES, PROCESSED_DIR, RAW_DIR
from utils.utils import get_sorting_keys

load_dotenv()


@dataclass
class DataGenerator:
    team_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    player_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    ts_lookup: dict = field(default_factory=dict)

    def __post_init__(self):
        self.s3_session = self.create_s3_session()
        self.oracle = OraclesElixir(
            session=self.s3_session, bucket=getenv("BUCKET_NAME")
        )
        self.imputer = EarlyGameStatsImputer()

    @staticmethod
    def create_s3_session():
        """
        Create a boto3 session to access the S3 bucket.
        """
        return boto3.Session(
            aws_access_key_id=getenv("ACCESS_ID"),
            aws_secret_access_key=getenv("SECRET_ID"),
        )

    @staticmethod
    def _get_years_to_process():
        """
        Get the years to process for the data ingestion.
        Those are the current year and the two previous years.
        """
        current_year = dt.date.today().year
        return [str(year) for year in range(current_year, current_year - 3, -1)]

    @staticmethod
    def _remove_buggy_games(data):
        """
        Remove games that have been identified as buggy.
        """
        with open(INVALID_GAMES, "r") as file:
            invalid_config = json.load(file)

        invalid_games = invalid_config["invalid_games"]
        data = data[~data.gameid.isin(invalid_games)].copy()
        logger.info("Removed buggy games.\n")
        return data

    def ingest_data_from_s3(self):
        """
        Ingest data from S3 bucket and store it in the interim directory.
        """
        try:
            # Define time frame for analytics
            years = self._get_years_to_process()

            # Ingest Data
            data = self.oracle.ingest_data(years=years)
            data.to_csv(RAW_DIR / "raw_data.csv", index=False)

            logger.info("Stored ingested data from S3.\n")

            # Remove Buggy Games
            data = self._remove_buggy_games(data)

            # Clean Data
            self.team_data = self.oracle.clean_data(data, split_on="team")
            self.player_data = self.oracle.clean_data(data, split_on="player")

            logger.info("Cleaned all data.\n")

            # Store Interim Data
            self.team_data.to_csv(INTERIM_DIR / "team_data.csv", index=False)
            self.player_data.to_csv(INTERIM_DIR / "player_data.csv", index=False)
            logger.info("Stored interim data.\n")
        except Exception as e:
            logger.error(f"Failed to ingest data from S3: {e}")
            raise

    def _enrich_data_with_ratings(self):
        """
        Enrich data with all the associated ratings.
        """
        logger.info("Enriching data with ratings...")

        self.player_data = Ratings.compute_elo(df=self.player_data, entity="player")
        self.team_data = Ratings.compute_elo(df=self.team_data, entity="team")
        logger.info("Enriched data with elo.")

        self.player_data = Ratings.compute_plackett_luce(df=self.player_data,
                                                         entity="player")
        self.team_data = Ratings.compute_plackett_luce(df=self.team_data,
                                                       entity="team")
        logger.info("Enriched data with plackett-luce.")

        self.player_data, self.team_data, self.ts_lookup = Ratings.compute_trueskill(
            player_data=self.player_data, team_data=self.team_data
        )
        logger.info("Enriched data with trueskill.")

        logger.info("Enriched data with ratings\n")

    def _enrich_data_with_performance_metrics(self):
        self.player_data = PerformanceMetrics.add_egpm_model(self.player_data,
                                                             entity="player")
        self.team_data = PerformanceMetrics.add_egpm_model(self.team_data,
                                                           entity="team")
        logger.info("Enriched data with EGPM.")

        self.player_data = PerformanceMetrics.add_entity_ema_statistics(
            self.player_data, entity="player"
        )
        self.team_data = PerformanceMetrics.add_entity_ema_statistics(
            self.team_data, entity="team"
        )
        logger.info("Enriched data with EMA statistics.")

        self.player_data = PerformanceMetrics.add_side_win_rate_ewm(
            self.player_data, entity="player"
        )
        self.team_data = PerformanceMetrics.add_side_win_rate_ewm(self.team_data,
                                                                  entity="team")
        logger.info("Enriched data with side win rate.")

        logger.info("Enriched player and team data with performance metrics.")

    def enrich_datasets(self):
        """
        Compute all enrichment for team and player-based analytics and predictions.
        """
        try:
            # Load Data
            self.team_data = pd.read_csv(INTERIM_DIR / "team_data.csv")
            self.player_data = pd.read_csv(INTERIM_DIR / "player_data.csv")

            # Impute missing data for players only
            self.player_data = self.imputer.process_data(self.player_data)

            # Enrich Data with Ratings
            self._enrich_data_with_ratings()

            # Enrich Data with player performance metrics
            self._enrich_data_with_performance_metrics()

            # Sort dfs before storing them
            self.player_data.sort_values(get_sorting_keys("player"), inplace=True)
            self.team_data.sort_values(get_sorting_keys("team"), inplace=True)

            # Store Enriched Data
            self.team_data.to_csv(PROCESSED_DIR / "team_data.csv", index=False)
            self.player_data.to_csv(PROCESSED_DIR / "player_data.csv", index=False)
            logger.info("Stored enriched data.\n")
        except Exception as e:
            logger.error(f"Failed to enrich dataset: {e}")
            raise

    def flatten_team_data(self):
        """
        Flatten the team_data dataframe to get the most recent record per team.
        """
        flattened_teams = (
            self.team_data.sort_values(["teamid", "date"])
            .groupby("teamid")
            .tail(1)
            .reset_index(drop=True)
        )
        flattened_teams = flattened_teams[
            [
                "date",
                "league",
                "teamname",
                "teamid",
                "elo",
                "pl_mu",
                "pl_sigma",
                "trueskill_sum_mu",
                "trueskill_sigma_squared",
                "egpm_dominance_ratio_ema_after",
                "ema_gamelength_after",
                "ema_gamelength_std_after",
                "ema_gamelength_growth_after",
                "ema_kills_after",
                "ema_kills_std_after",
                "ema_kills_growth_after",
                "ema_deaths_after",
                "ema_deaths_std_after",
                "ema_deaths_growth_after",
                "ema_assists_after",
                "ema_assists_std_after",
                "ema_assists_growth_after",
                "ema_kda_after",
                "ema_kda_std_after",
                "ema_kda_growth_after",
                "ema_goldat15_after",
                "ema_goldat15_std_after",
                "ema_goldat15_growth_after",
                "ema_xpat15_after",
                "ema_xpat15_std_after",
                "ema_xpat15_growth_after",
                "ema_csat15_after",
                "ema_csat15_std_after",
                "ema_csat15_growth_after",
                "ema_golddiffat15_after",
                "ema_golddiffat15_std_after",
                "ema_golddiffat15_growth_after",
                "ema_xpdiffat15_after",
                "ema_xpdiffat15_std_after",
                "ema_xpdiffat15_growth_after",
                "ema_csdiffat15_after",
                "ema_csdiffat15_std_after",
                "ema_csdiffat15_growth_after",
                "ema_egpm_after",
                "ema_egpm_std_after",
                "ema_egpm_growth_after",
                "ema_barons_after",
                "ema_barons_std_after",
                "ema_barons_growth_after",
                "ema_dragons_after",
                "ema_dragons_std_after",
                "ema_dragons_growth_after",
                "ema_towers_after",
                "ema_towers_std_after",
                "ema_towers_growth_after",
                "ema_firstblood_after",
                "ema_firstblood_std_after",
                "ema_firstblood_growth_after",
                "ema_red_side_after",
                "ema_blue_side_after",
                "ema_side_win_perc",
            ]
        ]
        flattened_teams = flattened_teams.rename(
            columns={
                'elo': 'team_elo',
                'pl_mu': 'team_pl_mu',
                'pl_sigma': 'team_pl_sigma',
                'trueskill_sum_mu': 'team_trueskill_sum_mu',
                'trueskill_sigma_squared': 'team_trueskill_sigma_squared',
                'egpm_dominance_ratio_ema_after': 'egpm_dominance_ratio',
                'ema_gamelength_after': 'gamelength',
                'ema_gamelength_std_after': 'gamelength_std',
                'ema_gamelength_growth_after': 'gamelength_growth',
                'ema_kills_after': 'kills',
                'ema_kills_std_after': 'kills_std',
                'ema_kills_growth_after': 'kills_growth',
                'ema_deaths_after': 'deaths',
                'ema_deaths_std_after': 'deaths_std',
                'ema_deaths_growth_after': 'deaths_growth',
                'ema_assists_after': 'assists',
                'ema_assists_std_after': 'assists_std',
                'ema_assists_growth_after': 'assists_growth',
                'ema_kda_after': 'kda',
                'ema_kda_std_after': 'kda_std',
                'ema_kda_growth_after': 'kda_growth',
                'ema_goldat15_after': 'goldat15',
                'ema_goldat15_std_after': 'goldat15_std',
                'ema_goldat15_growth_after': 'goldat15_growth',
                'ema_xpat15_after': 'xpat15',
                'ema_xpat15_std_after': 'xpat15_std',
                'ema_xpat15_growth_after': 'xpat15_growth',
                'ema_csat15_after': 'csat15',
                'ema_csat15_std_after': 'csat15_std',
                'ema_csat15_growth_after': 'csat15_growth',
                'ema_golddiffat15_after': 'golddiffat15',
                'ema_golddiffat15_std_after': 'golddiffat15_std',
                'ema_golddiffat15_growth_after': 'golddiffat15_growth',
                'ema_xpdiffat15_after': 'xpdiffat15',
                'ema_xpdiffat15_std_after': 'xpdiffat15_std',
                'ema_xpdiffat15_growth_after': 'xpdiffat15_growth',
                'ema_csdiffat15_after': 'csdiffat15',
                'ema_csdiffat15_std_after': 'csdiffat15_std',
                'ema_csdiffat15_growth_after': 'csdiffat15_growth',
                'ema_egpm_after': 'egpm',
                'ema_egpm_std_after': 'egpm_std',
                'ema_egpm_growth_after': 'egpm_growth',
                'ema_barons_after': 'barons',
                'ema_barons_std_after': 'barons_std',
                'ema_barons_growth_after': 'barons_growth',
                'ema_dragons_after': 'dragons',
                'ema_dragons_std_after': 'dragons_std',
                'ema_dragons_growth_after': 'dragons_growth',
                'ema_towers_after': 'towers',
                'ema_towers_std_after': 'towers_std',
                'ema_towers_growth_after': 'towers_growth',
                'ema_firstblood_after': 'firstblood',
                'ema_firstblood_std_after': 'firstblood_std',
                'ema_firstblood_growth_after': 'firstblood_growth',
                'ema_red_side_after': 'red_side',
                'ema_blue_side_after': 'blue_side',
                'ema_side_win_perc': 'side_win_perc'
            }
        )
        flattened_teams.to_csv(PROCESSED_DIR / "flattened_teams.csv", index=False)
        logger.info("Stored flattened teams data.")

    def flatten_player_data(self):
        """
        Flatten the player_data dataframe to get the most recent record per player per team.
        """
        flattened_players = (
            self.player_data.sort_values(["playerid", "date"])
            .groupby(["playerid", "teamid"])
            .tail(1)
            .reset_index(drop=True)
        )
        flattened_players = flattened_players[
            [
                "date",
                "position",
                "league",
                "playername",
                "playerid",
                "teamname",
                "teamid",
                "elo",
                "pl_mu",
                "pl_sigma",
                "trueskill_mu",
                "trueskill_sigma",
                "egpm_dominance_ratio_ema_after",
                "ema_gamelength_after",
                "ema_gamelength_std_after",
                "ema_gamelength_growth_after",
                "ema_kills_after",
                "ema_kills_std_after",
                "ema_kills_growth_after",
                "ema_deaths_after",
                "ema_deaths_std_after",
                "ema_deaths_growth_after",
                "ema_assists_after",
                "ema_assists_std_after",
                "ema_assists_growth_after",
                "ema_kda_after",
                "ema_kda_std_after",
                "ema_kda_growth_after",
                "ema_goldat15_after",
                "ema_goldat15_std_after",
                "ema_goldat15_growth_after",
                "ema_xpat15_after",
                "ema_xpat15_std_after",
                "ema_xpat15_growth_after",
                "ema_csat15_after",
                "ema_csat15_std_after",
                "ema_csat15_growth_after",
                "ema_golddiffat15_after",
                "ema_golddiffat15_std_after",
                "ema_golddiffat15_growth_after",
                "ema_xpdiffat15_after",
                "ema_xpdiffat15_std_after",
                "ema_xpdiffat15_growth_after",
                "ema_csdiffat15_after",
                "ema_csdiffat15_std_after",
                "ema_csdiffat15_growth_after",
                "ema_egpm_after",
                "ema_egpm_std_after",
                "ema_egpm_growth_after",
                "ema_ckpm_after",
                "ema_ckpm_std_after",
                "ema_ckpm_growth_after",
                "ema_kill_participation_after",
                "ema_kill_participation_std_after",
                "ema_kill_participation_growth_after",
                "ema_wcpm_after",
                "ema_wcpm_std_after",
                "ema_wcpm_growth_after",
                "ema_deathsat15_after",
                "ema_deathsat15_std_after",
                "ema_deathsat15_growth_after",
                "ema_vspm_after",
                "ema_vspm_std_after",
                "ema_vspm_growth_after",
                "ema_xp_efficiency_after",
                "ema_xp_efficiency_std_after",
                "ema_xp_efficiency_growth_after",
                "ema_cspm_after",
                "ema_cspm_std_after",
                "ema_cspm_growth_after",
                "ema_earnedgoldshare_after",
                "ema_earnedgoldshare_std_after",
                "ema_earnedgoldshare_growth_after",
                "ema_gold_efficiency_after",
                "ema_gold_efficiency_std_after",
                "ema_gold_efficiency_growth_after",
                "ema_wpm_after",
                "ema_wpm_std_after",
                "ema_wpm_growth_after",
                "ema_assistsat15_after",
                "ema_assistsat15_std_after",
                "ema_assistsat15_growth_after",
                "ema_damageshare_after",
                "ema_damageshare_std_after",
                "ema_damageshare_growth_after",
                "ema_dpm_after",
                "ema_dpm_std_after",
                "ema_dpm_growth_after",
                "ema_total_cs_after",
                "ema_total_cs_std_after",
                "ema_total_cs_growth_after",
                "ema_red_side_after",
                "ema_blue_side_after",
                "ema_side_win_perc",
            ]
        ]
        flattened_players = flattened_players.rename(
            columns={
                "elo": "player_elo",
                'pl_mu': 'player_pl_mu',
                'pl_sigma': 'player_pl_sigma',
                'trueskill_mu': 'player_trueskill_mu',
                'trueskill_sigma': 'player_trueskill_sigma',
                'egpm_dominance_ratio_ema_after': 'egpm_dominance_ratio',
                'egpm_opp_dominance_ratio_ema_after': 'egpm_opp_dominance_ratio',
                'ema_gamelength_after': 'gamelength',
                'ema_gamelength_std_after': 'gamelength_std',
                'ema_gamelength_growth_after': 'gamelength_growth',
                'ema_kills_after': 'kills',
                'ema_kills_std_after': 'kills_std',
                'ema_kills_growth_after': 'kills_growth',
                'ema_deaths_after': 'deaths',
                'ema_deaths_std_after': 'deaths_std',
                'ema_deaths_growth_after': 'deaths_growth',
                'ema_assists_after': 'assists',
                'ema_assists_std_after': 'assists_std',
                'ema_assists_growth_after': 'assists_growth',
                'ema_kda_after': 'kda',
                'ema_kda_std_after': 'kda_std',
                'ema_kda_growth_after': 'kda_growth',
                'ema_goldat15_after': 'goldat15',
                'ema_goldat15_std_after': 'goldat15_std',
                'ema_goldat15_growth_after': 'goldat15_growth',
                'ema_xpat15_after': 'xpat15',
                'ema_xpat15_std_after': 'xpat15_std',
                'ema_xpat15_growth_after': 'xpat15_growth',
                'ema_csat15_after': 'csat15',
                'ema_csat15_std_after': 'csat15_std',
                'ema_csat15_growth_after': 'csat15_growth',
                'ema_golddiffat15_after': 'golddiffat15',
                'ema_golddiffat15_std_after': 'golddiffat15_std',
                'ema_golddiffat15_growth_after': 'golddiffat15_growth',
                'ema_xpdiffat15_after': 'xpdiffat15',
                'ema_xpdiffat15_std_after': 'xpdiffat15_std',
                'ema_xpdiffat15_growth_after': 'xpdiffat15_growth',
                'ema_csdiffat15_after': 'csdiffat15',
                'ema_csdiffat15_std_after': 'csdiffat15_std',
                'ema_csdiffat15_growth_after': 'csdiffat15_growth',
                'ema_egpm_after': 'egpm',
                'ema_egpm_std_after': 'egpm_std',
                'ema_egpm_growth_after': 'egpm_growth',
                'ema_ckpm_after': 'ckpm',
                'ema_ckpm_std_after': 'ckpm_std',
                'ema_ckpm_growth_after': 'ckpm_growth',
                'ema_kill_participation_after': 'kill_participation',
                'ema_kill_participation_std_after': 'kill_participation_std',
                'ema_kill_participation_growth_after': 'kill_participation_growth',
                'ema_wcpm_after': 'wcpm',
                'ema_wcpm_std_after': 'wcpm_std',
                'ema_wcpm_growth_after': 'wcpm_growth',
                'ema_deathsat15_after': 'deathsat15',
                'ema_deathsat15_std_after': 'deathsat15_std',
                'ema_deathsat15_growth_after': 'deathsat15_growth',
                'ema_vspm_after': 'vspm',
                'ema_vspm_std_after': 'vspm_std',
                'ema_vspm_growth_after': 'vspm_growth',
                'ema_xp_efficiency_after': 'xp_efficiency',
                'ema_xp_efficiency_std_after': 'xp_efficiency_std',
                'ema_xp_efficiency_growth_after': 'xp_efficiency_growth',
                'ema_cspm_after': 'cspm',
                'ema_cspm_std_after': 'cspm_std',
                'ema_cspm_growth_after': 'cspm_growth',
                'ema_earnedgoldshare_after': 'earnedgoldshare',
                'ema_earnedgoldshare_std_after': 'earnedgoldshare_std',
                'ema_earnedgoldshare_growth_after': 'earnedgoldshare_growth',
                'ema_gold_efficiency_after': 'gold_efficiency',
                'ema_gold_efficiency_std_after': 'gold_efficiency_std',
                'ema_gold_efficiency_growth_after': 'gold_efficiency_growth',
                'ema_wpm_after': 'wpm',
                'ema_wpm_std_after': 'wpm_std',
                'ema_wpm_growth_after': 'wpm_growth',
                'ema_assistsat15_after': 'assistsat15',
                'ema_assistsat15_std_after': 'assistsat15_std',
                'ema_assistsat15_growth_after': 'assistsat15_growth',
                'ema_damageshare_after': 'damageshare',
                'ema_damageshare_std_after': 'damageshare_std',
                'ema_damageshare_growth_after': 'damageshare_growth',
                'ema_dpm_after': 'dpm',
                'ema_dpm_std_after': 'dpm_std',
                'ema_dpm_growth_after': 'dpm_growth',
                'ema_total_cs_after': 'total_cs',
                'ema_total_cs_std_after': 'total_cs_std',
                'ema_total_cs_growth_after': 'total_cs_growth',
                'ema_red_side_after': 'red_side',
                'ema_blue_side_after': 'blue_side',
                'ema_side_win_perc': 'side_win_perc'}
        )
        flattened_players[
            ["trueskill_mu", "trueskill_sigma"]] = flattened_players.apply(
            lambda row: [self.ts_lookup[row["playerid"]].mu,
                         self.ts_lookup[row["playerid"]].sigma]
            if row["playerid"] in self.ts_lookup else [None, None],
            axis=1,
            result_type="expand",
        )
        flattened_players.to_csv(PROCESSED_DIR / "flattened_players.csv", index=False)
        logger.info("Stored flattened players data.")

    def run(self):
        logger.info("Starting data generation.\n")
        self.ingest_data_from_s3()
        self.enrich_datasets()
        self.flatten_team_data()
        self.flatten_player_data()
        logger.info("Data generation completed.")


if __name__ == "__main__":
    generator = DataGenerator()
    logger.warning("Make sure to close any open CSV files!")

    start = dt.datetime.now()
    generator.run()
    end = dt.datetime.now()

    seconds_elapsed = (end - start).seconds
    logger.info(f"Data generation took {seconds_elapsed} seconds.\n")
