#!/bin/bash

set -eo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$DIR/.."

export DEVICE='cpu'

python -m src.train_zeshel_local --val-check-interval 1 --limit-train-batches=2 --batch-size=4
