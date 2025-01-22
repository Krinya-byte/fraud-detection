#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <lambda_function_name>"
  exit 1
fi

LAMBDA_FUNCTION_NAME=$1

mkdir -p ../build/lambdas/$LAMBDA_FUNCTION_NAME && cd ../build/lambdas/$LAMBDA_FUNCTION_NAME 
cp -r ../../../src/lambdas/$LAMBDA_FUNCTION_NAME/* .
python3 -m venv .venv
source .venv/bin/activate
if [ -e requirements.txt ]; then python3 -m pip install -r requirements.txt -t . --only-binary=:all:; fi
deactivate
rm -r .venv
ls -la
zip -r ../../target/$LAMBDA_FUNCTION_NAME.zip . 
rm -r ../$LAMBDA_FUNCTION_NAME  