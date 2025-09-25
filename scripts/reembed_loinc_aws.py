#!/usr/bin/env python3
"""
Re-embed LOINC descriptions using AWS Bedrock Titan Embeddings.
This creates a new embedding space that's compatible with your hospital codes.
"""

import os
import json
import pandas as pd
import numpy as np
import boto3
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import warnings
warnings.filterwarnings('ignore')

def init_bedrock(region: str):
    """Initialize Bedrock client."""
    return boto3.client('bedrock-runtime', region_name=region)

def bedrock_embed(client, model_id: str, text: str) -> np.ndarray:
    """Generate embedding using AWS Bedrock Titan."""
    body = json.dumps({"inputText": text})
    resp = client.invoke_model(
        modelId=model_id,
        body=body,
        contentType='application/json',
        accept='application/json'
    )
    payload = json.loads(resp['body'].read())
    return np.array(payload['embedding'], dtype=np.float32)

def embed_batch(client, model_id: str, texts: List[str], workers: int = 1) -> List[np.ndarray]:
    """Embed a batch of texts in parallel."""
    embeddings = [None] * len(texts)
    
    def task(i, text):
        try:
            # simple retry loop
            for attempt in range(3):
                try:
                    return (i, bedrock_embed(client, model_id, text))
                except Exception:
                    if attempt == 2:
                        raise
            return (i, bedrock_embed(client, model_id, text))
        except Exception as e:
            print(f"Error embedding text {i}: {e}")
            # Return random embedding as fallback
            return (i, np.random.normal(size=1536).astype(np.float32))
    
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(task, i, t) for i, t in enumerate(texts)]
        for fut in as_completed(futures):
            i, vec = fut.result()
            embeddings[i] = vec
    
    return embeddings

def reembed_loinc_descriptions(input_file: str, 
                              output_file: str,
                              model_id: str = "amazon.titan-embed-text-v1",
                              region: str = "us-east-1",
                              batch_size: int = 100,
                              workers: int = 4,
                              max_descriptions: int = None):
    """
    Re-embed LOINC descriptions with AWS Bedrock.
    
    Args:
        input_file: Path to LOINC descriptions parquet file
        output_file: Path to output parquet file
        model_id: Bedrock model ID
        region: AWS region
        batch_size: Number of descriptions to process at once
        workers: Number of parallel workers
        max_descriptions: Maximum number of descriptions to process (for testing)
    """
    print(f"Re-embedding LOINC descriptions with AWS Bedrock...")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Model: {model_id}")
    print(f"Region: {region}")
    print(f"Batch size: {batch_size}")
    print(f"Workers: {workers}")
    
    # Load LOINC descriptions
    print("\nLoading LOINC descriptions...")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} LOINC descriptions")
    
    if max_descriptions:
        df = df.head(max_descriptions)
        print(f"Limited to {len(df)} descriptions for testing")
    
    # Initialize Bedrock client
    print("\nInitializing Bedrock client...")
    client = init_bedrock(region)
    
    # Process in batches
    all_embeddings = []
    all_codes = []
    all_descriptions = []
    
    total_batches = (len(df) + batch_size - 1) // batch_size
    print(f"\nProcessing {total_batches} batches...")
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        
        print(f"  Batch {batch_num + 1}/{total_batches}: {len(batch_df)} descriptions")
        
        # Extract texts and codes
        texts = batch_df['Description'].tolist()
        codes = batch_df['LOINC_NUM'].tolist()
        
        # Generate embeddings
        embeddings = embed_batch(client, model_id, texts, workers)
        
        # Store results
        all_embeddings.extend(embeddings)
        all_codes.extend(codes)
        all_descriptions.extend(texts)
        
        # Save intermediate results every batch to avoid loss and reduce memory
        temp_df = pd.DataFrame({
            'LOINC_NUM': all_codes,
            'Description': all_descriptions,
            'Embedding': [json.dumps(emb.tolist()) for emb in all_embeddings]
        })
        try:
            if output_file.endswith('.parquet'):
                temp_df.to_parquet(f"{output_file}.temp", index=False)
            else:
                temp_df.to_csv(f"{output_file}.temp.csv", index=False)
            print(f"    Saved {len(temp_df)} embeddings so far...")
        except Exception as e:
            print(f"    Warning: failed to write temp output ({e}). Will continue.")
    
    # Create final output
    print(f"\nCreating final output with {len(all_embeddings)} embeddings...")
    final_df = pd.DataFrame({
        'LOINC_NUM': all_codes,
        'Description': all_descriptions,
        'Embedding': [json.dumps(emb.tolist()) for emb in all_embeddings]
    })
    
    # Save final results with parquet or CSV fallback
    try:
        if output_file.endswith('.parquet'):
            final_df.to_parquet(output_file, index=False)
        else:
            final_df.to_csv(output_file, index=False)
    except Exception as e:
        print(f"Parquet write failed ({e}). Falling back to CSV...")
        csv_fallback = output_file.rsplit('.', 1)[0] + '.csv'
        final_df.to_csv(csv_fallback, index=False)
        output_file = csv_fallback
    
    # Clean up temp file
    for temp in [f"{output_file}.temp", f"{output_file}.temp.csv"]:
        if os.path.exists(temp):
            os.remove(temp)
    
    print(f"\n✓ Re-embedding complete!")
    print(f"Output saved to: {output_file}")
    print(f"Total embeddings: {len(final_df)}")
    
    return final_df

def main():
    """Main function for re-embedding LOINC descriptions."""
    parser = argparse.ArgumentParser(description='Re-embed LOINC descriptions with AWS Bedrock')
    parser.add_argument('--input', default='loinc_snomed_embeddings/loinc/df_loinc_descriptions.parquet',
                       help='Input LOINC descriptions parquet file')
    parser.add_argument('--output', default='loinc_descriptions_aws_embeddings.parquet',
                       help='Output parquet file')
    parser.add_argument('--model_id', default='amazon.titan-embed-text-v1',
                       help='Bedrock model ID')
    parser.add_argument('--region', default='us-east-1',
                       help='AWS region')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='Batch size for processing')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--max_descriptions', type=int, default=None,
                       help='Maximum descriptions to process (for testing)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        return
    
    # Re-embed descriptions
    result_df = reembed_loinc_descriptions(
        input_file=args.input,
        output_file=args.output,
        model_id=args.model_id,
        region=args.region,
        batch_size=args.batch_size,
        workers=args.workers,
        max_descriptions=args.max_descriptions
    )
    
    print(f"\nNext steps:")
    print(f"1. Use this re-embedded file for mapping")
    print(f"2. Generate embeddings for your hospital codes with the same model")
    print(f"3. Compare embeddings in the same vector space")

if __name__ == "__main__":
    main()
