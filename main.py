# CIT5900-Research-Graph/main.py

import argparse
from code.metadata_extraction.nih import process_nih_records 
from code.metadata_extraction.openalex import process_openalex_records
from code.processing.deduplication import deduplicate_data
from code.visualization import generate_visualizations

def run_pipeline(extract_nih=True, extract_openalex=True):
    """Main pipeline execution function"""
    print("🚀 Starting FSRDC Publications Pipeline...")
    
    # Data Extraction
    if extract_nih:
        print("\n🔍 Extracting NIH PMC records...")
        process_nih_records()
    
    if extract_openalex:
        print("\n🔍 Extracting OpenAlex records...")
        process_openalex_records()
    
    # Data Processing
    print("\n⚙️ Processing data...")
    deduplicate_data()
    
    # Visualization
    print("\n📊 Generating visualizations...")
    generate_visualizations()
    
    print("\n✅ Pipeline completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-nih', action='store_true', help="Skip NIH extraction")
    parser.add_argument('--skip-openalex', action='store_true', help="Skip OpenAlex extraction")
    args = parser.parse_args()
    
    run_pipeline(
        extract_nih=not args.skip_nih,
        extract_openalex=not args.skip_openalex
    )