import pandas as pd


def calculate_performance(data: pd.DataFrame, alpha: float = 0.3) -> pd.DataFrame:
    # Convert date column to datetime
    data["date"] = pd.to_datetime(data["date"])

    # Sort values for data integrity & consistency
    sort_keys = ["date", "league", "gameid", "teamid", "position", "result"]
    data = data.sort_values(sort_keys).reset_index(drop=True)

    # Compute the EWMA for player-specific metrics
    metrics = [
        "kills",
        "deaths",
        "assists",
        "goldat15",
        "xpat15",
        "csat15",
        "golddiffat15",
        "xpdiffat15",
        "earnedgoldshare",
        "dpm",
        "damageshare",
        "wpm",
        "wcpm",
        "vspm",
        "ckpm",
        "cspm",
        "egpm",
    ]
    for metric in metrics:
        ewma_col_name = f"ewma_{metric}"
        data[ewma_col_name] = data.groupby("playerid")[metric].transform(
            lambda x: x.ewm(alpha=alpha).mean()
        )

    # KDA Ratio
    data["kda_ratio"] = (data["kills"] + data["assists"]) / data["deaths"].replace(0, 1)

    # Gold and XP Efficiency
    data["gold_efficiency"] = data["goldat15"] / (
        data["kills"] + data["assists"] + data["deaths"]
    )
    data["xp_efficiency"] = data["xpat15"] / (
        data["kills"] + data["assists"] + data["deaths"]
    )

    # Kill Participation
    team_kills = data.groupby(["gameid", "teamid"])["kills"].transform("sum")
    data["kill_participation"] = (data["kills"] + data["assists"]) / team_kills

    # Growth Metrics
    for metric in metrics:
        ewma_col_name = f"ewma_{metric}"
        growth_col_name = f"growth_{metric}"
        data[growth_col_name] = data[metric] - data[ewma_col_name]

    # Volatility Metrics
    for metric in metrics:
        volatility_col_name = f"volatility_{metric}"
        data[volatility_col_name] = data.groupby("playerid")[metric].transform(
            lambda x: x.rolling(window=5).std()
        )

    # Save processed data if needed
    output_path = "data/processed/player_performance.csv"
    data.to_csv(output_path, index=False)

    return data
