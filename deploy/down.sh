#!/usr/bin/env bash
# Tear down the Novus Support Agent stack (acmera-support-<env>).
# Run this before novus-rag-assistant/deploy/down.sh.
# ECR images and the app-keys secret are not touched.
set -euo pipefail

REGION="${AWS_REGION:-ap-south-1}"
ENVIRONMENT="${ENVIRONMENT:-dev}"
STACK="acmera-support-${ENVIRONMENT}"

echo "Deleting ${STACK} in ${REGION} (takes about 5 minutes) ..."
aws cloudformation delete-stack --region "$REGION" --stack-name "$STACK"
aws cloudformation wait stack-delete-complete --region "$REGION" --stack-name "$STACK"
echo "${STACK} deleted. Now run novus-rag-assistant/deploy/down.sh to stop the rest of the bill."
