#!/bin/bash

# This script needs to be run by Circle CI so all needed environment variables
# can be available.

set -eo pipefail

#env=$1  # 'dev' or 'prod'
env='test'
repo_name=docker  # the artifact registry repo name
location=us-central1  # region the artifact registry repo lives at
project_id="bavard-${env}"
version=1.0.0  # docker images' tag version
short_commit_sha=$(echo "$CIRCLE_SHA1" | cut -c -7)
model_name=entity_linker

# Configure GCP environment
echo "$GOOGLE_CREDENTIALS" | gcloud auth activate-service-account --key-file=-
gcloud config set project "$project_id"
gcloud beta auth configure-docker "${location}-docker.pkg.dev,gcr.io"

# Build and push docker containers

base_image="entity-linker-base"
# These images are pushed to GCR
training_image="gcr.io/$project_id/entity-linker-training"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$DIR/../dockerfiles"

# Build the images

docker build -t "$base_image" -f base.Dockerfile ..
docker build -t "$training_image:$version" -f training.Dockerfile ..

# Push the images

docker push "$training_image:$version"

cd ..  # go back to root

# Provision the AI Platform model resource for the model.
# It's just a container where the model versions will live.

if [[ $(gcloud ai-platform models list --region $location --filter $model_name) != *"$model_name"* ]]; then
  # $model_name is not found in the current models, so create it.
  gcloud beta ai-platform models create $model_name \
    --description "Embeds entities in their mentions in a shared space." \
    --enable-console-logging \
    --enable-logging \
    --region $location
fi

# Apply Terraform

#export TF_VAR_slack_webhook_url=$SLACK_WEBHOOK_DEV
#short_commit_sha=$(echo "$CIRCLE_SHA1" | cut -c -7)
#export TF_VAR_short_commit_sha=$short_commit_sha
#terraform/apply.sh "$env"