# 📌 IMPORTS
import os
import random
import csv
import gzip
import requests
import numpy as np
import pandas as pd

from sklearn.metrics import (
    matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score
)

import transformers
from transformers import (
    AutoTokenizer, Trainer, AutoModelForSequenceClassification, TrainingArguments, AutoModelForMaskedLM
)
from torch.utils.data import Dataset


# 📌 DOWNLOAD CLINVAR VCF FILE (ONLY IF NOT ALREADY DOWNLOADED)
def download_vcf(url, output_file):
    """Downloads a VCF file from ClinVar if it does not already exist."""
    if os.path.exists(output_file):
        print(f"✅ File already exists: {output_file}, skipping download.")
        return
    
    print(f"📥 Downloading {output_file}...")
    response = requests.get(url, stream=True)
    with open(output_file, "wb") as f:
        f.write(response.content)
    
    print(f"✅ File downloaded successfully: {output_file}")


# 📌 1️⃣ READ VARIANTS FROM A COMPRESSED `.vcf.gz` FILE
def read_vcf_gz(vcf_file):
    """Reads a compressed .vcf.gz file and extracts cancer variants with risk factor."""
    variants = []
    with gzip.open(vcf_file, "rt") as f:  # Open in text mode
        for line in f:
            if line.startswith("#"):  # Skip header lines
                continue
            fields = line.strip().split("\t")
            chrom, pos, ref, alt, info = fields[0], int(fields[1]), fields[3], fields[4].split(",")[0], fields[7]

            # Check if the variant has a risk factor in the CLNSIG field
            if "CLNSIG=" in info and "risk" in info.split("CLNSIG=")[1].split(";")[0]:
                variants.append((chrom, pos, ref, alt))
    
    return variants

# 📌 2️⃣ GET THE REFERENCE SEQUENCE AROUND THE VARIANT
def get_sequence(chrom, pos, flank=500):
    """Fetches the reference DNA sequence ±flank nucleotides around the variant."""
    start, end = max(1, pos - flank), pos + flank
    url = f"https://rest.ensembl.org/sequence/region/human/{chrom}:{start}-{end}?content-type=text/plain"

    response = requests.get(url)
    return (response.text.strip(), start, end) if response.status_code == 200 else (None, None, None)

# 📌 3️⃣ INSERT MUTATION INTO THE SEQUENCE
def mutate_sequence(sequence, ref, alt, flank=500):
    """Inserts mutation (SNP or Indel) into the reference DNA sequence."""
    center = flank  # Position of the variant in the extracted sequence
    if sequence[center:center + len(ref)] != ref:
        print(f"⚠ Mismatch: Expected {ref} but found {sequence[center:center + len(ref)]}")
        return None
    return sequence[:center] + alt + sequence[center + len(ref):]

# 📌 4️⃣ SAVE TO A TSV FILE
def write_tsv(output_file, data):
    """Writes variant data to a TSV file."""
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["chromosome", "start_of_bin", "end_of_bin", "label", "sequence"])
        writer.writerows(data)

# 📌 5️⃣ RUN THE FULL PIPELINE
def process_variants(vcf_file, output_tsv, flank_size=500, sample_size=0):
    """Full pipeline to generate mutated sequences in TSV format."""
    variants = read_vcf_gz(vcf_file)
    print(f"🔍 Found {len(variants)} variants with risk factor.")

    # Apply random sampling if requested
    if sample_size > 0:
        sample_size = min(sample_size, len(variants))
        variants = random.sample(variants, sample_size)
        print(f"🔍 Randomly selected {sample_size} variants.")

    output_data = []

    for chrom, pos, ref, alt in variants:
        print(f"📌 Processing variant {chrom}:{pos} {ref}>{alt}")

        # Get reference sequence
        seq, start, end = get_sequence(chrom, pos, flank_size)
        if not seq:
            continue  # Skip this variant if error occurs

        # Insert mutation
        mutated_seq = mutate_sequence(seq, ref, alt, flank_size)
        if not mutated_seq:
            continue  # Skip if mutation failed

        # Add both mutated and reference sequences
        output_data.append([chrom, start, end, 1, mutated_seq])
        output_data.append([chrom, start, end, 0, seq])

    # Write to TSV file
    write_tsv(output_tsv, output_data)
    print(f"✅ Cancer variant sequences with risk factor saved to {output_tsv}")


# 📌 LOAD PRE-TRAINED LLM MODEL
def load_grover_model(local_dir="./grover_local", model_name="PoetschLab/GROVER"):
    """Downloads and caches the GROVER model and tokenizer locally."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    tokenizer.save_pretrained(local_dir)
    model.save_pretrained(local_dir)
    print(f"✅ Model and tokenizer saved locally in {local_dir}")


VCF_FILE = "clinvar_cancer.vcf.gz"
VCF_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"
print( "Start")
download_vcf(VCF_URL, VCF_FILE)

# 📌 PARAMETERS
TSV_OUTPUT = "cancer_variants.tsv"
FLANK_SIZE = 500  # Number of nucleotides before/after the mutation
SAMPLE_SIZE = 200  # Number of variants to process (0 for all)

# 📌 RUN THE SCRIPT
process_variants(VCF_FILE, TSV_OUTPUT, FLANK_SIZE, SAMPLE_SIZE)


load_grover_model()
