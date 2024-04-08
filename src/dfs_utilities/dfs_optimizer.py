import json
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Union

import pandas as pd


@dataclass
class RosterOptimizer:
    dk: pd.DataFrame
    fade_list: List[str]
    salary_cap: int = 1500000
    max_team_size: int = 3

    def prep_dk_csv(self) -> Dict[str, List[str]]:
        """
        Prepare the DraftKings CSV data.
        Transforms raw data from DraftKings and formats it for the optimizer.

        Returns:
            dict: A dictionary containing match data.
        """
        self.dk["Name"] = self.dk["Name"].str.strip()
        self.dk = self.dk[~self.dk["Name"].isin(self.fade_list)]
        match_list = self._generate_match_list(self.dk["Game Info"].unique())
        return self._generate_match_evaluation_string(match_list)

    @staticmethod
    def _generate_match_list(raw_match_list: List[str]) -> List[str]:
        """
        Generate a list of matches from the raw match data.

        Args:
            raw_match_list (List[str]): Raw match list to process.
        Returns:
            List[str]: Processed match list.
        """
        match_list = []
        for match in raw_match_list:
            match = match.split(" ", 1)[0]
            match = match.replace("@", '" , "')
            match = '["' + match + '"]'
            match_list.append(match)
        return match_list

    @staticmethod
    def _generate_match_evaluation_string(
        match_list: List[str],
    ) -> Dict[str, List[str]]:
        """
        Generate a match evaluation string from the list of matches.

        Args:
            match_list (List[str]): List of matches.
        Returns:
            dict: Evaluation string for matches.
        """
        match_string = "{"
        for i, match in enumerate(match_list, start=1):
            string = f'"game{i}": {match},'
            match_string += string
        match_string = match_string.rstrip(",") + "}"
        return json.loads(match_string)

    def initialize_rosters(self) -> Dict[str, pd.DataFrame]:
        """
        Initialize roster dataframes based on positions.

        Returns:
            dict: Dictionary of dataframes for each position.
        """
        positions = ["Top", "Jng", "Mid", "ADC", "Sup", "Team"]
        rosters = {pos.lower(): self.dk[self.dk["Position"] == pos].reset_index(drop=True) for pos in positions}
        return rosters

    def generate_individual_roster(self, rosters: Dict[str, pd.DataFrame]) -> Dict[str, Union[str, int]]:
        """
        Generate an individual roster based on constraints.

        Args:
            rosters (Dict[str, pd.DataFrame]): Dictionary of dataframes
                                                for each position.
        Returns:
            dict: The best roster based on the given constraints.
        """
        best_roster = {
            "Top": "",
            "Jng": "",
            "Mid": "",
            "ADC": "",
            "Sup": "",
            "Team": "",
            "Cost": 0,
            "Score": 0,
        }

        # Using itertools product to simplify nested loops
        for top, jng, mid, adc, sup, team in product(
            rosters["top"].iterrows(),
            rosters["jng"].iterrows(),
            rosters["mid"].iterrows(),
            rosters["adc"].iterrows(),
            rosters["sup"].iterrows(),
            rosters["team"].iterrows(),
        ):
            _, top = top
            _, jng = jng
            _, mid = mid
            _, adc = adc
            _, sup = sup
            _, team = team

            ind_cost = top["Cost"] + jng["Cost"] + mid["Cost"] + adc["Cost"] + sup["Cost"] + team["Cost"]
            if ind_cost > self.salary_cap:
                continue

            ind_score = top["Pts"] + jng["Pts"] + mid["Pts"] + adc["Pts"] + sup["Pts"] + team["Pts"]
            if ind_score <= best_roster["Score"]:
                continue

            # Check team count constraint
            teams = [
                top["team"],
                jng["team"],
                mid["team"],
                adc["team"],
                sup["team"],
                team["team"],
            ]
            if max([teams.count(t) for t in set(teams)]) > self.max_team_size:
                continue

            best_roster = {
                "Top": f"{top['team']} {top['player']}",
                "Jng": f"{jng['team']} {jng['player']}",
                "Mid": f"{mid['team']} {mid['player']}",
                "ADC": f"{adc['team']} {adc['player']}",
                "Sup": f"{sup['team']} {sup['player']}",
                "Team": f"{team['team']} {team['player']}",
                "Cost": ind_cost,
                "Score": ind_score,
            }

        del best_roster["Score"]

        if all(value == "" or value == 0 for value in best_roster.values()):
            raise Exception("No mathematically possible rosters for this stack.")
        else:
            return best_roster

    def optimize(self) -> Dict[str, Union[str, int]]:
        """Optimize and generate the best possible roster.
        Returns:
            dict: Best possible roster based on constraints.
        """
        rosters = self.initialize_rosters()
        return self.generate_individual_roster(rosters)
