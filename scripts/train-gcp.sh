#!/bin/bash

set -eo pipefail

base_model_type="ROBERTA_BASE"

project_id="bavard-test-293219"
job_name="entity_linker_${base_model_type}_$(date +%m%d%H%M%S)"
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
  --batch-size=4 \
  --val-check-interval=200 \
  --max-epochs=5 \
  --base-model-type=base_model_type

echo "kicked off training job ${job_name}"
gcloud ai-platform jobs describe "$job_name"
