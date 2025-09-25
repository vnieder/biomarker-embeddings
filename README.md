# AWS Bedrock Hospital Code Mapping Project

This project maps proprietary hospital codes to standardized LOINC codes using AWS Bedrock Titan Embeddings with consistent vector space.

## Project Structure

```
aws_mapping_project/
├── src/                          # Source code
│   └── hospital_code_mapper_aws.py
├── scripts/                      # Utility scripts
│   └── reembed_loinc_aws.py
├── data/                         # Input data
│   ├── hospital_codes.csv        # Your hospital codes data
│   └── loinc_snomed_embeddings/  # Original LOINC/SNOMED embeddings
├── cache/                        # Cached embeddings
├── outputs/                      # Results
├── requirements.txt
├── run_test.sh                  # Quick test script
└── README.md
```

## Quick Start

### 1. Setup Environment

```bash
cd aws_mapping_project
pip install -r requirements.txt
```

### 2. Set AWS Credentials

```bash
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_SESSION_TOKEN="your_session_token"  # if using temporary credentials
export AWS_REGION="us-east-1"
```

### 3. Run Quick Test

```bash
./run_test.sh
```

This will:

- Re-embed 100 LOINC descriptions
- Test mapping on 50 hospital codes
- Show results and quality metrics

### 4. Scale Up (Optional)

```bash
# Re-embed 10K LOINC descriptions for better accuracy
python scripts/reembed_loinc_aws.py \
  --input data/loinc_snomed_embeddings/loinc/df_loinc_descriptions.parquet \
  --output cache/loinc_aws_embeddings_10k.parquet \
  --max_descriptions 10000 \
  --batch_size 16 \
  --workers 2

# Test mapping with better coverage
python src/hospital_code_mapper_aws.py \
  --loinc_file cache/loinc_aws_embeddings_10k.parquet \
  --input_csv data/your_hospital_codes.csv \
  --rows 500 \
  --topk 5
```

## Workflow

1. **Re-embed LOINC descriptions** with AWS Bedrock Titan (one-time cost)
2. **Generate embeddings** for your hospital codes with same model
3. **Compare embeddings** in the same vector space for accurate matching

## Key Benefits

- ✅ **Consistent embedding space** - both LOINC and hospital codes use AWS Bedrock
- ✅ **High accuracy** - proper semantic similarity matching (0.3-0.4+ scores)
- ✅ **AWS-native** - no OpenAI dependencies
- ✅ **Scalable** - handles large datasets efficiently
- ✅ **Memory-safe** - processes data in chunks to avoid crashes

## Performance Results

**Test Results with 10K LOINC embeddings:**

- Similarity scores: 0.32-0.44 (vs 0.1-0.25 with 100 embeddings)
- Medically relevant matches: Hospital codes properly matching related LOINC codes
- Semantic similarity working across different medical terminology

## Cost Considerations

- Re-embedding 100 LOINC descriptions: ~$0.50 (test)
- Re-embedding 10K LOINC descriptions: ~$10-20
- Re-embedding 56K LOINC descriptions: ~$50-100 (full dataset)
- Mapping hospital codes: ~$0.10 per 1000 codes

## Files

- `src/hospital_code_mapper_aws.py` - Main mapping script with CLI args
- `scripts/reembed_loinc_aws.py` - Re-embed LOINC descriptions with AWS Bedrock
- `run_test.sh` - Quick test script for validation
- `cache/` - Stores re-embedded LOINC data
- `outputs/` - Mapping results (CSV format)

## CLI Usage

```bash
# Re-embed LOINC descriptions
python scripts/reembed_loinc_aws.py \
  --input data/loinc_snomed_embeddings/loinc/df_loinc_descriptions.parquet \
  --output cache/loinc_aws_embeddings_10k.parquet \
  --max_descriptions 10000 \
  --batch_size 16 \
  --workers 2

# Map hospital codes
python src/hospital_code_mapper_aws.py \
  --loinc_file cache/loinc_aws_embeddings_10k.parquet \
  --input_csv data/your_hospital_codes.csv \
  --output_csv outputs/loinc_mappings_aws.csv \
  --rows 500 \
  --topk 5
```
