#!/bin/bash

# This script needs to be run by Circle CI so all needed environment variables
# can be available.

set -eo pipefail

repo_name=docker  # the artifact registry repo name
location=us-central1  # region the artifact registry repo lives at
project_id="bavard-test-293219"
version=1.0.0  # docker images' tag version
model_name=entity_linker

# Configure GCP environment
gcloud config set project "$project_id"
gcloud beta auth configure-docker "${location}-docker.pkg.dev,gcr.io"

# Build and push docker containers

base_image="entity-linker-base"
# These images are pushed to GCR
training_image="gcr.io/$project_id/entity-linker-training"
eval_image="gcr.io/$project_id/entity-linker-eval"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$DIR/../dockerfiles"

# Build the images

docker build -t "$base_image" -f base.Dockerfile ..
docker build -t "$training_image:$version" -f training.Dockerfile ..
docker build -t "$eval_image:$version" -f eval.Dockerfile ..

# Push the images

docker push "$training_image:$version"
docker push "$eval_image:$version"
