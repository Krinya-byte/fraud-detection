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

# variable "iam_role_name" {
#   description = "The name of the IAM role for the Lambda function"
#   type        = string
# }
