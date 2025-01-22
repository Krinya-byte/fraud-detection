import json
import boto3
import pandas as pd
from io import StringIO
import os

def lambda_handler(event, context):
    s3 = boto3.client('s3')
    
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        
        response = s3.get_object(Bucket=bucket, Key=key)
        csv_content = response['Body'].read().decode('utf-8')
        
        df = pd.read_csv(StringIO(csv_content))
        
        columns_to_drop = ['ownername', 'owneremail', 'dealershipaddress','saledate','iban']
        df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        
        df.dropna(subset=['CarName','color', 'fueltype','aspiration','carbody','drivewheel'], inplace=True)
        
        curated_bucket = os.environ['CURATED_BUCKET']
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        s3.put_object(Bucket=curated_bucket, Key=key, Body=csv_buffer.getvalue())
        
    return {
        'statusCode': 200,
        'body': json.dumps('Processing complete')
    }