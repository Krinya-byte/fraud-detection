variable "s3_landing_bucket_name" {
  description = "The name of the S3 bucket for landing CSV files"
  type        = string
}

variable "s3_curated_bucket_name" {
  description = "The name of the S3 bucket for curated data"
  type        = string
}

variable "lambda_function_name" {
  description = "The name of the AWS Lambda function"
  type        = string
}

variable "lambda_runtime" {
  description = "The runtime environment for the Lambda function"
  type        = string
  default     = "python3.8"
}

variable "lambda_memory_size" {
  description = "The amount of memory allocated to the Lambda function"
  type        = number
  default     = 128
}

variable "lambda_timeout" {
  description = "The timeout for the Lambda function in seconds"
  type        = number
  default     = 30
}

variable "sagemaker_domain_name" {
  description = "Name for the SageMaker Studio Domain"
  type        = string
  default     = "yettel-fraud-studio-domain"
}

variable "sagemaker_user_profile_name" {
  description = "Name for the SageMaker Studio User Profile"
  type        = string
  default     = "default-user"
}

variable "api_gateway_name" {
  description = "Name for the API Gateway"
  type        = string
  default     = "s3-upload-api"
}

variable "api_lambda_function_name" {
  description = "Name for the Lambda function that generates pre-signed URLs"
  type        = string
  default     = "s3-presigned-url-generator"
}

variable "cognito_user_pool_name" {
  description = "Name for the Cognito User Pool"
  type        = string
  default     = "yettel-fraud-detection-users"
}

variable "cognito_user_pool_client_name" {
  description = "Name for the Cognito User Pool Client"
  type        = string
  default     = "app-client"
}
