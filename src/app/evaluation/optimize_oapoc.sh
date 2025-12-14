#!/bin/bash
# Auto-generated optimization script for OAPOC pipeline

echo "========================================="
echo "OAPOC PIPELINE OPTIMIZATION"
echo "========================================="

# Navigate to project directory
cd ~/Desktop/research/poc/oapoc

# Backup current data
echo "Creating backup..."
mkdir -p backups/$(date +%Y%m%d)
cp -r src/data/processed backups/$(date +%Y%m%d)/
cp -r src/data/vector_store backups/$(date +%Y%m%d)/

# Update chunking parameters in normalizer.py
echo "Updating chunking parameters..."
sed -i 's/chunk_size=250/chunk_size=200/g' src/app/etl/normalizer.py
sed -i 's/overlap=50/overlap=100/g' src/app/etl/normalizer.py

# Clear old processed data
echo "Clearing old data..."
rm -f src/data/processed/*.json
rm -f src/data/vector_store/*

# Re-run pipeline with optimized parameters
echo "Re-processing data with optimized parameters..."
python src/app/main.py

echo "========================================="
echo "Optimization complete!"
echo "Run evaluation to verify improvements."
echo "========================================="
