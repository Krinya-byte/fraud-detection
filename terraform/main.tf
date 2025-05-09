terraform {
  required_providers {
    aws = {
      source  = "registry.terraform.io/hashicorp/aws"
      version = ">= 4.0" # Use a version that supports aws_iam_oidc_provider
    }
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0.1"
    }
  }
}

provider "docker" {
  registry_auth {
    address  = data.aws_ecr_authorization_token.token.proxy_endpoint
    username = data.aws_ecr_authorization_token.token.user_name
    password = data.aws_ecr_authorization_token.token.password
  }
}

data "aws_region" "current" {}

data "aws_ecr_authorization_token" "token" {}

resource "aws_s3_bucket" "landing_bucket" {
  bucket = var.s3_landing_bucket_name
}

resource "aws_s3_bucket" "curated_bucket" {
  bucket = var.s3_curated_bucket_name
}

# Reuse existing SageMaker Role for Studio
resource "aws_iam_role" "sagemaker_role" {
  name = "${var.sagemaker_domain_name}_execution_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Effect = "Allow"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_s3_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
  role       = aws_iam_role.sagemaker_role.name
}

resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
  role       = aws_iam_role.sagemaker_role.name
}

# Enable versioning for both buckets
resource "aws_s3_bucket_versioning" "landing_bucket_versioning" {
  bucket = aws_s3_bucket.landing_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_versioning" "curated_bucket_versioning" {
  bucket = aws_s3_bucket.curated_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Server-side encryption for both buckets
resource "aws_s3_bucket_server_side_encryption_configuration" "landing_bucket_encryption" {
  bucket = aws_s3_bucket.landing_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "curated_bucket_encryption" {
  bucket = aws_s3_bucket.curated_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Add lifecycle policy for S3 buckets
resource "aws_s3_bucket_lifecycle_configuration" "landing_bucket_lifecycle" {
  bucket = aws_s3_bucket.landing_bucket.id

  rule {
    id     = "cleanup_old_data"
    status = "Enabled"
    filter {
      prefix = ""
    }

    expiration {
      days = 30
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "curated_bucket_lifecycle" {
  bucket = aws_s3_bucket.curated_bucket.id

  rule {
    id     = "archive_old_data"
    status = "Enabled"
    filter {
      prefix = ""
    }
    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }
  }
}

# --- ECR Repository for Lambda Image ---

resource "aws_ecr_repository" "lambda_ecr_repo" {
  name = "${var.lambda_function_name}-repo" # You might want to customize this naming convention

  image_tag_mutability = "MUTABLE" # Or IMMUTABLE depending on your tagging strategy

  image_scanning_configuration {
    scan_on_push = true
  }
}

# --- Build and Push Lambda Docker Image ---
locals {
  # Define the path to the Dockerfile relative to the terraform directory
  lambda_source_path = "../src/lambdas/${var.lambda_function_name}"
  # Define the image name using the ECR repository URL
  image_name = "${aws_ecr_repository.lambda_ecr_repo.repository_url}:latest"
}

# Build the Docker image locally using the Docker provider
resource "docker_image" "lambda_image" {
  name = local.image_name
  build {
    context = abspath(local.lambda_source_path)
    # Dockerfile is expected to be in the context directory
    # Add build args if needed:
    # build_arg = {
    #   MY_ARG = "value"
    # }
  }
  # Ensure ECR repo exists before trying to build/tag with its name
  depends_on = [aws_ecr_repository.lambda_ecr_repo]
}

# Push the built image to ECR
resource "docker_registry_image" "lambda_ecr_image" {
  name          = docker_image.lambda_image.name
  keep_remotely = false # Set to true if you want to prevent Terraform from deleting the image from ECR on destroy

  # Implicit dependency on docker_image ensures it's built first
  # Explicit dependency on the token ensures auth is ready
  depends_on = [data.aws_ecr_authorization_token.token]
}

# --- CSV Preprocessor Lambda Function ---

resource "aws_iam_role" "lambda_role" {
  name = "${var.lambda_function_name}_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
        Effect = "Allow"
      },
    ]
  })
}

resource "aws_iam_policy" "lambda_policy" {
  name        = "${var.lambda_function_name}_policy"
  description = "Policy for Lambda function to access S3 buckets"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.landing_bucket.arn,
          "${aws_s3_bucket.landing_bucket.arn}/*",
          aws_s3_bucket.curated_bucket.arn,
          "${aws_s3_bucket.curated_bucket.arn}/*"
        ]
        Effect = "Allow"
      },
      {
        Action   = "logs:*"
        Resource = "*"
        Effect   = "Allow"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_policy_attachment" {
  policy_arn = aws_iam_policy.lambda_policy.arn
  role       = aws_iam_role.lambda_role.name
}

# Build package for the CSV preprocessor Lambda
# resource "null_resource" "package_build_csv_preprocessor" {
#   triggers = {
#     dir_sha1 = sha1(join("", [for f in fileset("${local.lambda_function_sources_path}/${var.lambda_function_name}", "*") : filesha1("${local.lambda_function_sources_path}/${var.lambda_function_name}/${f}")]))
#   }

#   provisioner "local-exec" {
#     # Execute using WSL, passing module root path
#     command     = "wsl bash ./../src/utils/package_lambda.sh ${var.lambda_function_name}"
#     working_dir = path.module
#   }
# }

resource "aws_lambda_function" "csv_preprocessor" {
  function_name = var.lambda_function_name
  role          = aws_iam_role.lambda_role.arn
  # handler       = "lambda_function.lambda_handler" # Defined in Dockerfile CMD
  # runtime       = "python3.12" # Defined in Dockerfile FROM
  timeout      = 60   # Increased timeout for data processing
  memory_size  = 1024 # Increased memory for data processing
  package_type = "Image"
  image_uri    = docker_registry_image.lambda_ecr_image.name # Use the name from the pushed image

  # filename = "${path.module}/../build/target/${var.lambda_function_name}.zip" # Replaced by image_uri

  environment {
    variables = {
      CURATED_BUCKET = aws_s3_bucket.curated_bucket.id,
    }
  }

  # depends_on = [null_resource.package_build_csv_preprocessor] # Removed as packaging is handled outside terraform apply
  depends_on = [docker_registry_image.lambda_ecr_image] # Ensure image is pushed first
}

resource "aws_lambda_permission" "landing_bucket" {
  statement_id  = "AllowExecutionFromS3Bucket"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.csv_preprocessor.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.landing_bucket.arn
}

resource "aws_s3_bucket_notification" "bucket_notification" {
  bucket = aws_s3_bucket.landing_bucket.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.csv_preprocessor.arn
    events              = ["s3:ObjectCreated:*"]
  }

  depends_on = [aws_lambda_permission.landing_bucket]
}

# --- Upload Dataset ---

resource "aws_s3_object" "dataset_upload" {
  bucket = aws_s3_bucket.landing_bucket.id
  key    = "dataset.txt"             # The desired name of the file in the S3 bucket
  source = abspath("../dataset.txt") # Path to the local file relative to the terraform module

  # Calculate etag based on file content for change detection
  etag = filemd5(abspath("../dataset.txt"))

  # Ensure the bucket exists before trying to upload
  depends_on = [aws_s3_bucket.landing_bucket]
}

# --- SageMaker Studio ---

# Data source for default VPC
data "aws_vpc" "default" {
  default = true
}

# Data source for subnets in the default VPC
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# Create SageMaker Studio Domain
resource "aws_sagemaker_domain" "studio_domain" {
  domain_name = var.sagemaker_domain_name
  auth_mode   = "IAM" # Use IAM for authentication
  vpc_id      = data.aws_vpc.default.id
  subnet_ids  = data.aws_subnets.default.ids

  default_user_settings {
    execution_role = aws_iam_role.sagemaker_role.arn
  }

  tags = {
    Name    = var.sagemaker_domain_name
    Project = "fraud_detection"
  }
}

# Create SageMaker Studio User Profile
resource "aws_sagemaker_user_profile" "studio_user" {
  domain_id         = aws_sagemaker_domain.studio_domain.id
  user_profile_name = var.sagemaker_user_profile_name

  user_settings {
    execution_role = aws_iam_role.sagemaker_role.arn
  }

  tags = {
    Name    = var.sagemaker_user_profile_name
    Project = "fraud_detection"
  }
}
