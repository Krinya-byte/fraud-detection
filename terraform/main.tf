
resource "aws_s3_bucket" "landing_bucket" {
  bucket = var.s3_landing_bucket_name
}

resource "aws_s3_bucket" "curated_bucket" {
  bucket = var.s3_curated_bucket_name
}

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

resource "null_resource" "package_build" {
 #triggered only when the sha1 of the directory changes
 triggers = {
    dir_sha1 = sha1(join("", [for f in fileset("${local.lambda_function_sources_path}/${var.lambda_function_name}", "*"): filesha1("${local.lambda_function_sources_path}/${var.lambda_function_name}/${f}")]))
  }

  #for installing packages and zipping the lambda function
  provisioner "local-exec" {
    command     = <<-EOT
    ./../utils/package_lambda.sh ${var.lambda_function_name}
    EOT
    working_dir = path.module
  }

}

resource "aws_lambda_function" "this" {
  function_name = var.lambda_function_name
  role          = aws_iam_role.lambda_role.arn
  handler       = "lambda_function.lambda_handler"
  runtime       = "python3.12"
  timeout       = 30

  filename = "${path.module}/../build/target/${var.lambda_function_name}.zip"

  environment {
    variables = {
      CURATED_BUCKET = aws_s3_bucket.curated_bucket.id
    }
  }
  depends_on = [null_resource.package_build] # triggered only when zip file is created

  lifecycle {
    replace_triggered_by = [ null_resource.package_build ] #deploy the lambda function when the zip file is updated
  }
}

resource "aws_lambda_permission" "landing_bucket" {
  statement_id  = "AllowExecutionFromS3Bucket"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.this.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.landing_bucket.arn
}

resource "aws_s3_bucket_notification" "bucket_notification" {
  bucket = aws_s3_bucket.landing_bucket.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.this.arn
    events              = ["s3:ObjectCreated:*"]
  }
  depends_on = [aws_lambda_permission.landing_bucket, aws_lambda_function.this]
}
