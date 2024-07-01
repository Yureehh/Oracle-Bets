import json

import boto3
from botocore.exceptions import ClientError


def get_secret(secret_name: str, region_name: str = "us-east-1") -> str:
    """
    Fetches a secret from AWS Secrets Manager.

    Args:
        secret_name (str): The name of the secret.
        region_name (str): The AWS region where the secret is stored.

    Returns:
        str: The secret value.
    """
    # Create a Secrets Manager client using the "lol_oracle" profile
    session = boto3.Session(profile_name="lol_oracle")
    client = session.client(service_name="secretsmanager", region_name=region_name)

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        print("Error retrieving secret:", e)
        raise

    # Parse and return the secret value
    if "SecretString" in get_secret_value_response:
        secret = get_secret_value_response["SecretString"]
    else:
        secret = get_secret_value_response["SecretBinary"]

    return secret


def get_secret_value(secret_name: str, key: str, region_name: str = "us-east-1") -> str:
    """
    Fetches a specific key from a secret stored in AWS Secrets Manager.

    Args:
        secret_name (str): The name of the secret.
        key (str): The key to retrieve from the secret.
        region_name (str): The AWS region where the secret is stored.

    Returns:
        str: The secret value.
    """
    secret = get_secret(secret_name, region_name)
    secret_dict = json.loads(secret)
    return secret_dict[key]
