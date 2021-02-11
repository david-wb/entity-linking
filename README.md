# Entity Linking

This repo contains a basic system for performing named entity linking. The ML models are implemented in PyTorch.

# Get the data

Download the training data from [here](https://drive.google.com/file/d/1X7ArrhJJQurRowweabLmhnK0pjvfzV9j/view?usp=sharing).

Then run 
```
tar -xzvf zeshel.tar.bz2
```

# Setup

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Run `conda env create -f environment.yml` from inside the extracted directory.
    - This creates a Conda environment called `enli`
3. Run `source activate enli`
