#!/bin/bash

set -eo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$DIR/.."

checkpoint_path="checkpoints/epoch=2-val_loss=0.06.ckpt"
data_dir="transformed_zeshel"

export DEVICE='cpu'

python -m src.compute_embeddings \
  --checkpoint-path "${checkpoint_path}" \
  --data-dir "${data_dir}"

