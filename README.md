# Entity Linking

This repo contains a basic system for performing named entity linking. The ML models are implemented in PyTorch.



# Setup

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Run `conda env create -f environment.yml` from inside the extracted directory.
    - This creates a Conda environment called `enli`
3. Run `source activate enli`

# Get the data

Download the training data from [here](https://drive.google.com/file/d/1X7ArrhJJQurRowweabLmhnK0pjvfzV9j/view?usp=sharing). 

Copy the downloaded file into the root folder of this repo and then run 
```
tar -xvf zeshel.tar.bz2
```

## Transform the Data
This step will require at least 20gb of memory.
```python
python transform_zeshel.py
```
