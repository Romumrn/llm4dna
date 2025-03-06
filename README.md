## Large Language Models for DNA Analysis

This repository is dedicated to exploring the application of Large Language Models (LLMs) in the analysis of DNA sequences using various available models.

For more information on the use of LLMs in genomics, please refer to the following presentation:

[https://slides-ai-dna-eff0eb.pages.in2p3.fr/#/title-slide](https://slides-ai-dna-eff0eb.pages.in2p3.fr/#/title-slide)

## GROVER Model Test

GROVER is a DNA language model designed to learn the sequence context within the human genome. It treats DNA sequences as a language, capturing the grammar, syntax, and semantics inherent in genomic data. This model has demonstrated proficiency in predicting subsequent DNA sequences and extracting biologically meaningful information, such as identifying gene promoters and protein binding sites. Notably, GROVER has also been shown to learn epigenetic processes—regulatory mechanisms that occur on top of the DNA sequence rather than being encoded within it.
link : [HuggingFace](https://huggingface.co/PoetschLab/GROVER)

To evaluate GROVER's capabilities, I utilized the tutorial notebook available at Zenodo:

[https://zenodo.org/records/13315363](https://zenodo.org/records/13315363)

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
Then execute the python pre training 
and open a slurm node to run the pyhton training script. 
http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-exec_interactif-eng.html


## Evo2 Model Test

https://github.com/arcinstitute/evo2

I want to test the use of evo on jeanzay to generate a sequence, to check the feasabilty. 

module load miniforge 
module load cuda
set the cuda variable with which nvcc

(not just need to load the good module)
Frist I will creat a new conda enviroenment 
(pyhton 11 because of the compatbility) 
conda create -n evo  python=3.11
conda activate evo
git clone https://github.com/ArcInstitute/evo2.git
cd evo2
pip install .

(pytorch-gpu-2.0.1+py3.10.12) [ulg68gj@jean-zay3: evo2]$ module list
Currently Loaded Modulefiles:
 1) cuda/11.7.1                4) gcc/8.5.0(8.3.1:8.4.1)   7) magma/2.7.0-cuda  10) libjpeg-turbo/2.1.3    
 2) nccl/2.12.12-1-cuda        5) openmpi/4.1.1-cuda       8) sox/14.4.2        11) ffmpeg/4.2.2           
 3) cudnn/8.5.0.96-11.7-cuda   6) intel-mkl/2020.4         9) sparsehash/2.0.3  12) pytorch-gpu/py3/2.0.1  

command: 

```
module load pytorch-gpu/py3/2.5.0 
Loading pytorch-gpu/py3/2.5.0
  Loading requirement: gcc/10.1.0 cuda/12.2.0 nccl/2.21.5-1-cuda cudnn/8.9.7.29-cuda openmpi/4.1.5-cuda intel-mkl/2020.4
    magma/2.7.2-cuda sox/14.4.2 hdf5/1.12.0-mpi-cuda libjpeg-turbo/2.1.3 ffmpeg/6.1.1 graphviz/2.49.0
(pytorch-gpu-2.5.0+py3.12.7) [ulg68gj@jean-zay3: ~]$ cd $WORK
(pytorch-gpu-2.5.0+py3.12.7) [ulg68gj@jean-zay3: ulg68gj]$ cd evo2/
(pytorch-gpu-2.5.0+py3.12.7) [ulg68gj@jean-zay3: evo2]$ pip install .
```
leads to an issue

```
        × python setup.py egg_info did not run successfully.
        │ exit code: 1
        ╰─> [9 lines of output]
            Traceback (most recent call last):
              File "<string>", line 2, in <module>
              File "<pip-setuptools-caller>", line 34, in <module>
              File "/tmp/pip-install-2pe8cl72/transformer-engine-torch_6db1de72f3b746cabcd7c636d851208f/setup.py", line 15, in <module>
                from torch.utils.cpp_extension import BuildExtension
              File "/linkhome/rech/genidl01/ulg68gj/.local/lib/python3.12/site-packages/torch/__init__.py", line 367, in <module>
                from torch._C import *  # noqa: F403
                ^^^^^^^^^^^^^^^^^^^^^^
            ImportError: /linkhome/rech/genidl01/ulg68gj/.local/lib/python3.12/site-packages/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4, version libnvJitLink.so.12
            [end of output]
```

I think that it's an issue with the version of python and the version of pythroch

### Test NT

Availiable of hugging face and github though JAX
https://huggingface.co/InstaDeepAI
https://github.com/instadeepai/nucleotide-transformer

Clone the repository to your local machine.
Install the package by running pip install .

work but OSError: [Errno 122] Disk quota exceeded