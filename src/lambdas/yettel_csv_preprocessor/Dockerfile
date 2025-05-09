
FROM public.ecr.aws/lambda/python:3.12

WORKDIR ${LAMBDA_TASK_ROOT}

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY yettel_csv_preprocessor.py .

CMD ["yettel_csv_preprocessor.lambda_handler"]
