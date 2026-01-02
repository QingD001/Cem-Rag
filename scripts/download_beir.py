#!/usr/bin/env python3
"""
Download BEIR datasets: msmarco, trec-covid, and nq
"""

from beir import util
import os

# Create data directory if not exists
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Datasets to download
datasets = ["msmarco", "trec-covid", "nq"]

print("Starting BEIR dataset downloads...")
print(f"Data will be saved to: {os.path.abspath(data_dir)}\n")

for dataset in datasets:
    print(f"Downloading {dataset}...")
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = os.path.join(data_dir, dataset)
    
    try:
        data_path = util.download_and_unzip(url, out_dir)
        print(f"✓ {dataset} downloaded and extracted to: {data_path}\n")
    except Exception as e:
        print(f"✗ Error downloading {dataset}: {e}\n")

print("Download complete!")


