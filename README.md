## Large Language Models for DNA Analysis

This repository explores the application of Large Language Models (LLMs) in analyzing DNA sequences using various models.

For more information on the use of LLMs in genomics, please refer to the following presentation:

[https://slides-ai-dna-eff0eb.pages.in2p3.fr/#/title-slide](https://slides-ai-dna-eff0eb.pages.in2p3.fr/#/title-slide)

## GROVER Model Evaluation

GROVER is a DNA language model designed to learn the sequence context within the human genome. It treats DNA sequences as a language, capturing the grammar, syntax, and semantics inherent in genomic data. This model has demonstrated proficiency in predicting subsequent DNA sequences and extracting biologically meaningful information, such as identifying gene promoters and protein-binding sites. Notably, GROVER has also been shown to learn epigenetic processes—regulatory mechanisms that occur on top of the DNA sequence rather than being encoded within it. 

The pre-trained GROVER model is available on Hugging Face: [https://huggingface.co/PoetschLab/GROVER](https://huggingface.co/PoetschLab/GROVER)

To evaluate GROVER's capabilities, I utilized the tutorial notebook available on Zenodo:

[https://zenodo.org/records/13135894](https://zenodo.org/records/13135894)

In this evaluation, I made a slight modification: instead of focusing on CTCF motifs, I aimed to detect whether a given sequence has the potential to be oncogenic. While I am uncertain about the suitability of this dataset for such an analysis, my primary objective was to assess the model's adaptability to different datasets.

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

Subsequently, I executed the Python preprocessing script and initiated a SLURM node to run the Python training script, as detailed here:

[http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-exec_interactif-eng.html](http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-exec_interactif-eng.html)

## Evo 2 Model Evaluation

Evo 2 is a genomic foundation model capable of generalist prediction and design tasks across DNA, RNA, and proteins. It utilizes a frontier deep learning architecture to model biological sequences at single-nucleotide resolution with near-linear scaling of compute and memory relative to context length. Evo 2 was trained with 40 billion parameters and a 1 megabase context length on over 9 trillion nucleotides from diverse eukaryotic and prokaryotic genomes. 

The Evo 2 repository is available on GitHub: [https://github.com/ArcInstitute/evo2](https://github.com/ArcInstitute/evo2)

I aimed to test the use of Evo 2 on Jean Zay to generate sequences and assess feasibility. The following steps were undertaken:

1. **Load Modules**:

   ```bash
   module load pytorch-gpu/py3/2.4.0
   module list
   1) cuda/12.2.0 4) gcc/10.1.0 7) magma/2.7.2-cuda 10)
   libjpeg-turbo/2.1.3
   2) nccl/2.19.3-1-cuda 5) openmpi/4.1.5-cuda 8) sox/14.4.2 11)
   ffmpeg/6.1.1
   3) cudnn/8.9.5.30-cuda 6) intel-mkl/2020.4 9) hdf5/1.12.0-mpi-cuda 12)
   pytorch-gpu/py3/2.4.0

2. **Clone and Install Evo 2**:

   ```bash
   git clone https://github.com/ArcInstitute/evo2.git
   cd evo2
   pip install .
   ```

However, I encountered an installation issue related to the `transformer-engine` package:

```
ImportError: /linkhome/rech/genidl01/ulg68gj/.local/lib/python3.12/site-packages/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4, version libnvJitLink.so.12
```

**Identified Issues**:

1. **Hardware Requirements**: Evo 2 utilizes Transformer Engine FP8 for some layers, which requires an H100 GPU (or other GPU with compute capability ≥8.9). https://github.com/ArcInstitute/evo2

2. **Compatibility**: The installation encountered problems due to the `vortex` submodule, which contains a Makefile not compatible with Jean Zay's environment.

## Nucleotide Transformer (NT) Model Evaluation

The Nucleotide Transformer (NT) is available on Hugging Face and GitHub through JAX:

- Hugging Face: [https://huggingface.co/InstaDeepAI](https://huggingface.co/InstaDeepAI)
- GitHub: [https://github.com/instadeepai/nucleotide-transformer](https://github.com/instadeepai/nucleotide-transformer)

I cloned the repository and attempted to install the package using `pip install .`. However, I encountered an `OSError: [Errno 122] Disk quota exceeded`, indicating insufficient disk space for downloading models. 
