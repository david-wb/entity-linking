#!/bin/bash

set -eo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$DIR/.."

checkpoint_path="checkpoints/epoch=0-step=4.ckpt"
data_dir="transformed_zeshel"

export DEVICE='cpu'

python -m src.eval_zeshel \
  --checkpoint-path "${checkpoint_path}" \
  --data-dir "${data_dir}"

