#!/bin/bash

set -eo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$DIR/.."

region="us-central1"
bucket="bavard-test-datasets"
model_dir="entity-linker"
data_file="bavard/zeshel.tar.bz2"

gcloud config set project "bavard-test-293219"

export DEVICE='cpu'

python -m src.train_zeshel_gcp \
  --job-dir "gs://${bucket}/${model_dir}" \
  --data-file "gs://${bucket}/${data_file}" \
  --batch-size=2 \
  --limit-train-batches=10 \
  --val-check-interval=10 \
  --max-epochs=1
