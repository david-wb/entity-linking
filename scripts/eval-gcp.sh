#!/bin/bash

set -eo pipefail

project_id="bavard-test-293219"
job_name="entity_linker_eval_$(date +%m%d%H%M%S)"
region="us-central1"
image="gcr.io/${project_id}/entity-linker-eval:1.0.0"

gcloud config set project "$project_id"

gcloud ai-platform jobs submit training "$job_name" \
  --region $region \
  --master-image-uri "$image" \
  --scale-tier custom \
  --master-machine-type n1-standard-8 \
  --master-accelerator count=1,type=nvidia-tesla-t4 \
  -- \
  --data-file "gs://bavard-test-datasets/bavard/zeshel.tar.bz2" \
  --checkpoint-path "gs://bavard-test-datasets/entity_linker_model/epoch=4-val_loss=0.053__BERT_BASE_03_21_1853_49.ckpt" \
  --base-model-type="BERT_BASE"

echo "kicked off training job ${job_name}"
gcloud ai-platform jobs describe "$job_name"
