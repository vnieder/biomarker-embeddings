# AWS Bedrock Hospital Code Mapping Project

This project maps proprietary hospital codes to standardized LOINC/SNOMED codes using AWS Bedrock Titan Embeddings.

## Project Structure

```
aws_mapping_project/
├── src/                          # Source code
│   └── hospital_code_mapper_aws.py
├── scripts/                      # Utility scripts
│   └── reembed_loinc_aws.py
├── data/                         # Input data
│   ├── Synthetic_Biomarker_Data_Part_1.csv
│   ├── Synthetic_Demographic_Data.csv
│   └── loinc_snomed_embeddings/  # Original embeddings
├── cache/                        # Cached embeddings
├── outputs/                      # Results
├── requirements.txt
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

### 3. Re-embed LOINC Descriptions (One-time setup)

```bash
# Test with small subset first
python scripts/reembed_loinc_aws.py --max_descriptions 1000 --output cache/loinc_aws_embeddings_test.parquet

# Full re-embedding (takes time and costs money)
python scripts/reembed_loinc_aws.py --output cache/loinc_aws_embeddings_full.parquet
```

### 4. Map Hospital Codes

```bash
# Test mapping with re-embedded LOINC data
python src/hospital_code_mapper_aws.py
```

## Workflow

1. **Re-embed LOINC descriptions** with AWS Bedrock Titan (one-time cost)
2. **Generate embeddings** for your hospital codes with same model
3. **Compare embeddings** in the same vector space for accurate matching

## Key Benefits

- ✅ **Consistent embedding space** - both LOINC and hospital codes use AWS Bedrock
- ✅ **High accuracy** - proper semantic similarity matching
- ✅ **AWS-native** - no OpenAI dependencies
- ✅ **Scalable** - handles large datasets efficiently

## Cost Considerations

- Re-embedding 56K LOINC descriptions: ~$50-100 (one-time)
- Mapping 1M hospital codes: ~$200-500 (per run)
- Use `--max_descriptions` for testing with smaller subsets

## Files

- `src/hospital_code_mapper_aws.py` - Main mapping script
- `scripts/reembed_loinc_aws.py` - Re-embed LOINC descriptions
- `cache/` - Stores re-embedded LOINC data
- `outputs/` - Mapping results
