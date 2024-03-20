from io import StringIO

import boto3


def create_s3_bucket(session, bucket_name):
    s3 = session.resource("s3")
    s3.create_bucket(Bucket=bucket_name)


def upload_data_to_s3(session, bucket_name, file_name, data):
    s3 = session.resource("s3")
    csv_buffer = StringIO()
    data.to_csv(csv_buffer, index=False)
    s3.Object(bucket_name, file_name).put(Body=csv_buffer.getvalue())


def get_s3_session(region_name):
    return boto3.Session(region_name=region_name)
