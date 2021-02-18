FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

WORKDIR /app

# google cloud sdk (needing so GCP can capture the logs)
RUN apt-get update -y \
    && apt-get install -y apt-utils \
    && apt-get install -y curl apt-transport-https ca-certificates gnupg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - \
    && apt-get update -y \
    && apt-get install google-cloud-sdk -y

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN wandb login 22e8bd21b54e97ed931e461f2eb039e08a7f01f2
COPY ./src ./src