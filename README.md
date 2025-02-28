## Large Language Models for DNA Analysis

This repository is dedicated to exploring the application of Large Language Models (LLMs) in the analysis of DNA sequences using various available models.

For more information on the use of LLMs in genomics, please refer to the following presentation:

[https://slides-ai-dna-eff0eb.pages.in2p3.fr/#/title-slide](https://slides-ai-dna-eff0eb.pages.in2p3.fr/#/title-slide)

## GROVER Model Evaluation

GROVER is a DNA language model designed to learn the sequence context within the human genome. It treats DNA sequences as a language, capturing the grammar, syntax, and semantics inherent in genomic data. This model has demonstrated proficiency in predicting subsequent DNA sequences and extracting biologically meaningful information, such as identifying gene promoters and protein binding sites. Notably, GROVER has also been shown to learn epigenetic processes—regulatory mechanisms that occur on top of the DNA sequence rather than being encoded within it. citeturn0search0

To evaluate GROVER's capabilities, I utilized the tutorial notebook available at Zenodo:

[https://zenodo.org/records/13315363](https://zenodo.org/records/13315363)

In this evaluation, I made a slight modification: instead of focusing on CTCF motifs, I aimed to detect whether a given sequence has the potential to be oncogenic.

Running the notebook directly on a Jean Zay computing node posed challenges, as these nodes lack internet access. To address this, I divided the process into two distinct parts:

1. **Preprocessing**: This phase involves downloading the necessary data and creating the dataset.

2. **Training**: This phase focuses on fine-tuning the model with the prepared dataset.

Following the guidelines provided here:

[http://www.idris.fr/jean-zay/gpu/jean-zay-gpu-python-env.html](http://www.idris.fr/jean-zay/gpu/jean-zay-gpu-python-env.html)

I relocated the Conda repositories to the `$WORK` directory instead of `$HOME` due to memory constraints.

To set up the environment, I executed the following commands:

```bash
conda env create -f env.yml
conda activate dna_llm_env
```
Then execute the python pre training 
and open a slurm node to run the pyhton training script. 