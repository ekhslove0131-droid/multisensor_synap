#!/usr/bin/env bash
set -euo pipefail

: "${CLOUD_RUN_SERVICE:=multisensor-goal15-onnx}"
: "${CLOUD_RUN_REGION:=asia-northeast3}"
: "${CLOUD_RUN_PROJECT:?Set CLOUD_RUN_PROJECT explicitly}"
: "${CLOUD_RUN_REPOSITORY:=cloud-run-source-deploy}"
: "${CLOUD_RUN_IMAGE_TAG:=goal15-onnx-v1}"

if [[ "${ALLOW_CLOUD_RUN_DEPLOY:-0}" != "1" ]]; then
  echo "Refusing deployment. Set ALLOW_CLOUD_RUN_DEPLOY=1 after image and hash review." >&2
  exit 2
fi

image="${CLOUD_RUN_REGION}-docker.pkg.dev/${CLOUD_RUN_PROJECT}/${CLOUD_RUN_REPOSITORY}/${CLOUD_RUN_SERVICE}:${CLOUD_RUN_IMAGE_TAG}"

# Build from the repository root: the Dockerfile copies the validated Python
# package and both model bundles from their repository paths. Using
# --source services/onnx_api would silently omit src/ and fail at build time.
gcloud builds submit \
  --project "${CLOUD_RUN_PROJECT}" \
  --tag "${image}" \
  --file services/onnx_api/Dockerfile \
  .

gcloud run deploy "${CLOUD_RUN_SERVICE}" \
  --project "${CLOUD_RUN_PROJECT}" \
  --region "${CLOUD_RUN_REGION}" \
  --image "${image}" \
  --port 8080 \
  --cpu 2 \
  --memory 2Gi \
  --no-allow-unauthenticated \
  --set-env-vars MODEL_ROOT=/app/models/goal15-final-v1
