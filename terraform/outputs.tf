output "landing_bucket_name" {
  description = "Name of the landing S3 bucket"
  value       = aws_s3_bucket.landing_bucket.id
}

output "curated_bucket_name" {
  description = "Name of the curated S3 bucket"
  value       = aws_s3_bucket.curated_bucket.id
}

output "sagemaker_studio_domain_url" {
  description = "URL for the SageMaker Studio domain"
  value       = aws_sagemaker_domain.studio_domain.url
}

output "lambda_function_name" {
  description = "Name of the CSV preprocessor Lambda function"
  value       = aws_lambda_function.csv_preprocessor.function_name
}

# Commenting out GitHub Actions role ARN output
# output "github_actions_role_arn" {
#   description = "ARN of the IAM role for GitHub Actions OIDC"
#   value       = aws_iam_role.github_actions_role.arn
# }
