# Bi-Encoder Entity Linking

This repo implements a bi-encoder model for entity linking. The bi-encoder separately embeds mention and entity
pairs into a shared vector space. The encoders in the bi-encoder model are pretrained transformers. 
We evaluate three different base encoder models on the retrieval rate metric.
The retrieval rate is the rate at which the correct entity for a mention is included when generating
`k` candidates for each mention in the test set.
The HuggingFace names of the three 
base encoder models are:

* `bert-base-uncased`
* `roberta-base`
* `johngiorgi/declutr-base`

The ML models in this repo are implemented using PyTorch and PyTorch-Lightning.



# Setup

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Run `conda env create -f environment.yml` from inside the extracted directory. 
   This creates a Conda environment called `enli`
3. Run 
   ```
   source activate enli
4. Install requirements.
   ```bash
   pip install -r requirements.txt
   ```

# Get the data

Download the training data from [here](https://drive.google.com/file/d/1X7ArrhJJQurRowweabLmhnK0pjvfzV9j/view?usp=sharing). 

Copy the downloaded file into the root folder of this repo and then run 
```
tar -xvf zeshel.tar.bz2
```

## Transform the Data
This step will require at least 20gb of memory.
```python
python -m src.transform_zeshel --input-dir="./zeshel"
```

## Training
To train on Google Cloud (GCP), you can edit `scripts/train-gcp.sh` with your own
google cloud project and the run
```bash
./scripts/train-gcp.sh
```
to submit a training job.

## Evaluation
Similarly, edit `scripts/eval-gcp.sh` with your google cloud project id and run
```bash
./scripts/eval-gcp.sh
```
to submit the eval job.
