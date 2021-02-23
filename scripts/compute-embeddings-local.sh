#!/bin/bash

set -eo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$DIR/.."

checkpoint_path="checkpoints/entity_linker_0222210142.ckpt"
data_dir="transformed_zeshel"

export DEVICE='cpu'

python -m src.compute_embeddings \
  --checkpoint-path "${checkpoint_path}" \
  --data-dir "${data_dir}"

