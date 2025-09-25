#!/usr/bin/env python3
"""
Hospital Code Mapper using AWS Bedrock Titan Embeddings.
This version uses re-embedded LOINC descriptions for consistent vector space.
"""

import pandas as pd
import numpy as np
import os
import json
from typing import List, Tuple, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
import boto3
import warnings
import argparse
warnings.filterwarnings('ignore')

class HospitalCodeMapperAWS:
    def __init__(self, 
                 loinc_embeddings_file: str = "loinc_descriptions_aws_embeddings.parquet",
                 aws_region: str = "us-east-1", 
                 model_id: str = "amazon.titan-embed-text-v1"):
        """
        Initialize mapper with AWS Bedrock and re-embedded LOINC data.
        
        Args:
            loinc_embeddings_file: Path to re-embedded LOINC descriptions
            aws_region: AWS region for Bedrock
            model_id: Bedrock model ID for embeddings
        """
        self.loinc_embeddings_file = loinc_embeddings_file
        self.aws_region = aws_region
        self.model_id = model_id
        self.bedrock_client = None
        self.loinc_embeddings = None
        self.loinc_descriptions = None
        
    def load_loinc_embeddings(self):
        """Load re-embedded LOINC descriptions."""
        if not os.path.exists(self.loinc_embeddings_file):
            raise FileNotFoundError(f"LOINC embeddings file not found: {self.loinc_embeddings_file}")
        
        print("Loading re-embedded LOINC descriptions...")
        self.loinc_embeddings = pd.read_parquet(self.loinc_embeddings_file)
        print(f"Loaded {len(self.loinc_embeddings)} LOINC embeddings")
        
        # Extract descriptions for reference
        self.loinc_descriptions = self.loinc_embeddings[['LOINC_NUM', 'Description']].copy()
        
    def initialize_bedrock_client(self):
        """Initialize AWS Bedrock client."""
        try:
            self.bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=self.aws_region
            )
            print(f"✓ Bedrock client initialized for region: {self.aws_region}")
                
        except Exception as e:
            print(f"✗ Error initializing Bedrock client: {e}")
            print("Set environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, AWS_REGION")
            raise
    
    def generate_embedding_for_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for input text using AWS Bedrock.
        
        Args:
            text: Input text to embed
            
        Returns:
            numpy array of embedding
        """
        if self.bedrock_client is None:
            self.initialize_bedrock_client()
        
        try:
            # Prepare the request body for Titan Embeddings
            body = json.dumps({
                "inputText": text
            })
            
            # Call Bedrock API
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType='application/json',
                accept='application/json'
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            embedding = np.array(response_body['embedding'])
            
            return embedding
            
        except Exception as e:
            print(f"Error generating embedding for text: '{text[:50]}...': {e}")
            # Return a random embedding as fallback for testing
            return np.random.rand(1536)
    
    def parse_embedding_from_json(self, embedding_json: str) -> np.ndarray:
        """Parse embedding from JSON string."""
        return np.array(json.loads(embedding_json), dtype=np.float32)
    
    def find_best_matches(self, 
                         query_text: str, 
                         top_k: int = 10,
                         max_samples: int = 1000) -> List[Tuple[str, str, float]]:
        """
        Find best matches for query text using cosine similarity.
        
        Args:
            query_text: Text to match
            top_k: Number of top matches to return
            max_samples: Maximum number of embeddings to process (for memory)
            
        Returns:
            List of (code, description, similarity_score) tuples
        """
        # Generate embedding for query text
        query_embedding = self.generate_embedding_for_text(query_text)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Sample embeddings to avoid memory issues
        if len(self.loinc_embeddings) > max_samples:
            sample_df = self.loinc_embeddings.sample(n=max_samples, random_state=42)
            print(f"  Sampling {max_samples} embeddings from {len(self.loinc_embeddings)} total")
        else:
            sample_df = self.loinc_embeddings
        
        # Parse embeddings from target data
        target_emb_list = []
        target_codes = []
        target_descriptions = []
        
        for _, row in sample_df.iterrows():
            try:
                embedding = self.parse_embedding_from_json(row['Embedding'])
                target_emb_list.append(embedding)
                target_codes.append(row['LOINC_NUM'])
                target_descriptions.append(row['Description'])
            except Exception as e:
                print(f"Error parsing embedding for {row['LOINC_NUM']}: {e}")
                continue
        
        if not target_emb_list:
            return []
        
        # Calculate cosine similarities
        target_embeddings_array = np.array(target_emb_list)
        similarities = cosine_similarity(query_embedding, target_embeddings_array)[0]
        
        # Get top k matches
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            code = target_codes[idx]
            description = target_descriptions[idx]
            similarity = similarities[idx]
            
            results.append((code, description, similarity))
        
        return results
    
    def map_hospital_codes(self, 
                          biomarker_data: pd.DataFrame,
                          code_column: str = 'code',
                          display_column: str = 'code_display',
                          top_k: int = 5) -> pd.DataFrame:
        """
        Map hospital codes to LOINC codes.
        
        Args:
            biomarker_data: DataFrame with hospital codes
            code_column: Column name containing codes
            display_column: Column name containing code descriptions
            top_k: Number of top matches to return
            
        Returns:
            DataFrame with mapping results
        """
        results = []
        
        print(f"Processing {len(biomarker_data)} hospital codes...")
        
        for idx, row in biomarker_data.iterrows():
            if idx % 10 == 0:
                print(f"  Processing code {idx+1}/{len(biomarker_data)}")
            
            code = row[code_column]
            display = row[display_column]
            
            # Find matches
            matches = self.find_best_matches(display, top_k)
            
            for i, (matched_code, description, similarity) in enumerate(matches):
                results.append({
                    'original_code': code,
                    'original_display': display,
                    'matched_code': matched_code,
                    'matched_description': description,
                    'similarity_score': similarity,
                    'rank': i + 1,
                    'target_standard': 'LOINC'
                })
        
        return pd.DataFrame(results)
    
    def analyze_mapping_quality(self, mapping_results: pd.DataFrame) -> Dict:
        """Analyze the quality of mapping results."""
        total_mappings = len(mapping_results)
        high_confidence = len(mapping_results[mapping_results['similarity_score'] > 0.8])
        medium_confidence = len(mapping_results[
            (mapping_results['similarity_score'] > 0.6) & 
            (mapping_results['similarity_score'] <= 0.8)
        ])
        low_confidence = len(mapping_results[mapping_results['similarity_score'] <= 0.6])
        
        return {
            'total_mappings': total_mappings,
            'high_confidence': high_confidence,
            'medium_confidence': medium_confidence,
            'low_confidence': low_confidence,
            'high_confidence_pct': (high_confidence / total_mappings * 100) if total_mappings > 0 else 0
        }

def main():
    """Example usage of the HospitalCodeMapperAWS."""
    print("AWS Bedrock Hospital Code Mapping")
    print("=" * 40)

    parser = argparse.ArgumentParser()
    parser.add_argument('--loinc_file', default='cache/loinc_aws_embeddings_test.parquet', help='Path to re-embedded LOINC parquet file')
    parser.add_argument('--region', default='us-east-1')
    parser.add_argument('--model_id', default='amazon.titan-embed-text-v1')
    parser.add_argument('--input_csv', default='data/Synthetic_Biomarker_Data_Part_1.csv')
    parser.add_argument('--output_csv', default='outputs/loinc_mappings_aws.csv')
    parser.add_argument('--rows', type=int, default=50, help='Number of input rows to process for test')
    parser.add_argument('--topk', type=int, default=3)
    args = parser.parse_args()

    # Initialize mapper
    mapper = HospitalCodeMapperAWS(
        loinc_embeddings_file=args.loinc_file,
        aws_region=args.region,
        model_id=args.model_id
    )
    
    # Load re-embedded LOINC data
    try:
        mapper.load_loinc_embeddings()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run: python scripts/reembed_loinc_aws.py --max_descriptions 1000 --output cache/loinc_aws_embeddings_test.parquet")
        return
    
    # Load your biomarker data (first 50 rows for testing)
    biomarker_data = pd.read_csv(args.input_csv).head(args.rows)
    
    print(f"\nProcessing {len(biomarker_data)} hospital codes...")
    
    # Map hospital codes to LOINC
    loinc_mappings = mapper.map_hospital_codes(biomarker_data, top_k=args.topk)
    
    # Analyze results
    quality = mapper.analyze_mapping_quality(loinc_mappings)
    
    print("\nMapping Quality:")
    print(f"High confidence (>0.8): {quality['high_confidence']} ({quality['high_confidence_pct']:.1f}%)")
    print(f"Medium confidence (0.6-0.8): {quality['medium_confidence']}")
    print(f"Low confidence (<0.6): {quality['low_confidence']}")
    
    # Save results
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    loinc_mappings.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")
    
    # Show sample results
    print("\nSample mappings:")
    for _, row in loinc_mappings.head(10).iterrows():
        print(f"  {row['original_code']} -> {row['matched_code']} (score: {row['similarity_score']:.3f})")

if __name__ == "__main__":
    main()
