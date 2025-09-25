#!/bin/bash
# Test script for AWS Bedrock Hospital Code Mapping

echo "AWS Bedrock Hospital Code Mapping - Test Run"
echo "============================================="

# Check if AWS credentials are set
if [ -z "$AWS_ACCESS_KEY_ID" ]; then
    echo "Error: AWS_ACCESS_KEY_ID not set"
    echo "Please run: export AWS_ACCESS_KEY_ID='your_key'"
    exit 1
fi

if [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "Error: AWS_SECRET_ACCESS_KEY not set"
    echo "Please run: export AWS_SECRET_ACCESS_KEY='your_secret'"
    exit 1
fi

echo "✓ AWS credentials found"

# Limit BLAS threads to reduce memory/segfault risk
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Step 1: Re-embed LOINC descriptions (small test)
echo ""
echo "Step 1: Re-embedding LOINC descriptions (test with 100 descriptions)..."
python scripts/reembed_loinc_aws.py \
    --input data/loinc_snomed_embeddings/loinc/df_loinc_descriptions.parquet \
    --output cache/loinc_aws_embeddings_test.parquet \
    --max_descriptions 100 \
    --batch_size 10 \
    --workers 1

if [ $? -ne 0 ]; then
    echo "Error: Failed to re-embed LOINC descriptions"
    exit 1
fi

echo "✓ LOINC re-embedding complete"

# Step 2: Test hospital code mapping
echo ""
echo "Step 2: Testing hospital code mapping..."
python src/hospital_code_mapper_aws.py \
  --loinc_file cache/loinc_aws_embeddings_test.parquet \
  --input_csv data/Synthetic_Biomarker_Data_Part_1.csv \
  --output_csv outputs/loinc_mappings_aws.csv \
  --rows 50 \
  --topk 3

if [ $? -ne 0 ]; then
    echo "Error: Failed to map hospital codes"
    exit 1
fi

echo ""
echo "✓ Test complete!"
echo "Results saved to: outputs/"
echo ""
echo "Next steps:"
echo "1. Review results in outputs/"
echo "2. If satisfied, run full re-embedding without --max_descriptions"
echo "3. Run full mapping on your complete dataset"
