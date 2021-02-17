#!/bin/bash

set -eo pipefail

# env=$1  # 'dev' or 'prod'
env='test'

short_commit_sha=$(echo "$CIRCLE_SHA1" | cut -c -7)
job_name=entity_linker_${short_commit_sha}
region="us-central1"
image="gcr.io/bavard-${env}/entity-linker-training:1.0.0"
bucket="bavard-${env}-entity-linker"
model_dir="entity-linker-model"
project_id="bavard-${env}"

echo "$GOOGLE_CREDENTIALS" | gcloud auth activate-service-account --key-file=-
gcloud config set project "$project_id"

gcloud ai-platform jobs submit training "$job_name" \
  --region $region \
  --master-image-uri "$image" \
  --scale-tier custom \
  --master-machine-type n1-standard-8 \
  --master-accelerator count=1,type=nvidia-tesla-t4 \
  --job-dir "gs://${bucket}/${model_dir}" \
  -- \
  --save_steps 1000000 \
  --num_train_epochs 4 \
  --disable_tqdm True \
  --model-version "$short_commit_sha" \
  --gcp-project-id "$project_id"

echo "kicked off training job ${job_name}"
gcloud ai-platform jobs describe "$job_name"