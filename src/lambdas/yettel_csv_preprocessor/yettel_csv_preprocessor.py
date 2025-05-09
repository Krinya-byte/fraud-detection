import os
import json
import pandas as pd
import boto3
from io import BytesIO, StringIO

def preprocess_data(df):
   df_cp = df.copy(deep=True)

    #rename cols to lower case
    df_cp.columns = df_cp.columns.str.lower()

    #drop unnecesarry string cols
    #convert to day time
    df_cp.day_id_encr = pd.to_datetime(df_cp.day_id_encr, format='%Y. %m. %d.', errors='coerce')
    df_cp['year'] = df_cp['day_id_encr'].dt.year
    df_cp['month'] = df_cp['day_id_encr'].dt.month
    df_cp['day'] = df_cp['day_id_encr'].dt.day
    df_cp = df_cp.drop(columns=['day_id_encr'])
    # Type Conversion and filling NaNs
    numeric_object_cols = df_cp.select_dtypes(include=np.float64)
    for col in numeric_object_cols:
        df_cp[col] = pd.to_numeric(df_cp[col], errors='coerce')
        if df_cp[col].isnull().any():
            median_val = df_cp[col].median()
            df_cp[col] = df_cp[col].fillna(median_val)

    #filling up the age column
    df_cp.r_age_y = pd.to_numeric(df_cp.r_age_y, errors='coerce')

    # Imputation
    categorical_cols = df_cp.select_dtypes(include=['object'])
    for col in categorical_cols:
        if col in df_cp.columns:
            if df_cp[col].isnull().any():
                 df_cp[col] = df_cp[col].fillna('Unknown')
   
    #Boolean convertion
    df_cp.fraud_status_6month = df_cp.fraud_status_6month.apply(lambda x: 1 if x == 'Y' else 0) 
    df_cp.instalment_ind = df_cp.instalment_ind.apply(lambda x: 1 if x =='Y' else 0)

    return df_cp

def lambda_handler(event, context):
    """
    Lambda function to preprocess CSV files uploaded to S3
    """
    s3 = boto3.client('s3')
    
    # Get bucket and file details from the event
    source_bucket = event['Records'][0]['s3']['bucket']['name']
    source_key = event['Records'][0]['s3']['object']['key']
    
    try:
        # Read CSV file from S3
        response = s3.get_object(Bucket=source_bucket, Key=source_key)

        csv_content = response['Body'].read().decode('utf-8') 
        df = pd.read_csv(BytesIO(csv_content),
                         delimiter=';',decimal=',', quoting=csv.QUOTE_NONE, encoding='utf-8-sig',encoding_errors='ignore', 
                         date_parser=lambda x: pd.to_datetime(x, format='%Y.%m.%d.'),
                         error_bad_lines=False,
                         warn_bad_lines=True)
        
        # Preprocess data
        processed_df = preprocess_data(df)
        
        # Save processed file to curated bucket
        curated_bucket = os.environ['CURATED_BUCKET']
        output_key = f"processed/{os.path.basename(source_key)}"
        
        csv_buffer = StringIO()
        processed_df.to_csv(csv_buffer, index=False)
        
        s3.put_object(
            Bucket=curated_bucket,
            Key=output_key,
            Body=csv_buffer.getvalue()
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'File processed successfully',
                'source_file': source_key,
                'output_file': output_key
            })
        }
        
    except Exception as e:
        print(f"Error processing file {source_key}: {str(e)}")
        raise 