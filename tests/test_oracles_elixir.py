from io import StringIO

import boto3
import pandas as pd
from moto import mock_s3

from src.data_ingest.oracles_elixir import OraclesElixir


@mock_s3
def test_ingest_data():
    # Create a boto3 session and resource after the mock has started
    session = boto3.Session(region_name="us-east-1")
    s3 = session.resource("s3")

    # Create the bucket
    s3.create_bucket(Bucket="test_bucket")

    # Create some test data and upload it to the mock S3 bucket
    test_data = pd.DataFrame(
        {
            "date": ["2022-01-01", "2022-01-02"],
            "gameid": ["game1", "game2"],
        }
    )
    csv_buffer = StringIO()
    test_data.to_csv(csv_buffer, index=False)
    s3.Object("test_bucket", "2022_LoL_esports_match_data_from_OraclesElixir.csv").put(
        Body=csv_buffer.getvalue()
    )

    # Create the OraclesElixir instance
    oe = OraclesElixir(session=session, bucket="test_bucket")

    # Call the method under test
    oe_data = oe.ingest_data([2022])

    # Check that the returned DataFrame matches the test data
    assert oe_data.equals(test_data)


def test_format_data_types():
    # Create the OraclesElixir instance
    oe = OraclesElixir(session=None, bucket=None)

    # Create some test data
    oe.oe_data = pd.DataFrame(
        {
            "date": ["2022-01-01", "2022-01-02"],
            "gameid": [" game1 ", " game2 "],
            "playerid": ["test1", "test2"],
            "teamid": ["team_a", "team_b"],
            "position": [" top ", " jungle "],
            # Add other columns as necessary...
        }
    )

    # Call the method under test
    oe.format_data_types()

    # Check that the 'date' column has been converted to datetime
    assert pd.api.types.is_datetime64_any_dtype(oe.oe_data["date"])
