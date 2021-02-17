#!/bin/bash

set -eo pipefail

project_id="bavard-test-293219"
short_commit_sha=$(echo "$CIRCLE_SHA1" | cut -c -7)
job_name=entity_linker_${short_commit_sha}
region="us-central1"
image="gcr.io/${project_id}/entity-linker-training:1.0.0"

gcloud config set project "$project_id"

gcloud ai-platform jobs submit training "$job_name" \
  --region $region \
  --master-image-uri "$image" \
  --scale-tier custom \
  --master-machine-type n1-standard-8 \
  --master-accelerator count=1,type=nvidia-tesla-t4 \
  -- \
  --data-file "gs://bavard-test-datasets/bavard/zeshel.tar.bz2" \
  --job-dir "gs://bavard-test-datasets/entity_linker_model" \

echo "kicked off training job ${job_name}"
gcloud ai-platform jobs describe "$job_name"
