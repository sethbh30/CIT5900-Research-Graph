import pandas as pd
import requests
import re
from time import sleep, time
from datetime import datetime, timedelta
import sys
import logging
from typing import Dict, List, Optional, Union
import concurrent.futures
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.CRITICAL)

# Configuration
CORE_API_URL = "https://api.core.ac.uk/v3/search/works"
CORE_API_KEY = "JitbKoV5gnCsEOXGWywqeA9zdNcvZSul"  # Replace with your own if needed
REQUEST_DELAY = 0.1  # Reduced delay between API calls
MAX_RETRIES = 5
RETRY_BACKOFF = [1, 2, 4, 8, 15]
MAX_WORKERS = 10  # Adjust based on your network and API rate limits

# FSRDC keywords (expanded list) - compiled for faster matching
FSRDC_KEYWORDS = [
    "census bureau", "fsrdc", "federal statistical research data center",
    "restricted microdata", "irs", "bea", "confidentiality review",
    "michigan rdc", "texas rdc", "boston rdc", "cornell rdc", "berkeley rdc",
    "annual survey of manufactures", "census of construction industries",
    "census of finance, insurance, and real estate", "ncrdc", "nyc rdc",
    "longitudinal employer-household dynamics", "lehd", "ces", "qwi",
    "confidential data", "restricted data", "statistical disclosure limitation",
    "title 13", "title 26", "cbms", "sbo", "economic census"
]

# Pre-compile regex patterns for faster matching
FSRDC_PATTERN = re.compile('|'.join(map(re.escape, FSRDC_KEYWORDS)))

# RDC mapping for faster lookups
RDC_MAPPING = {
    'michigan': 'Michigan RDC',
    'texas': 'Texas RDC',
    'boston': 'Boston RDC',
    'cornell': 'Cornell RDC',
    'berkeley': 'Berkeley RDC',
    'new york': 'NYC RDC',
    'north carolina': 'NCRDC'
}

# Session object for connection pooling
session = requests.Session()

def safe_api_call(url: str, headers: Optional[Dict] = None, params: Optional[Dict] = None) -> Optional[Dict]:
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, headers=headers, params=params, timeout=30)
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                sleep(retry_after)
                continue
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                logging.debug(f"API call failed: {str(e)}")
                return None
            sleep(RETRY_BACKOFF[attempt])
    return None

@lru_cache(maxsize=1024)
def clean_year(year: Union[str, int, float]) -> str:
    if pd.isna(year):
        return ''
    try:
        year_str = str(int(float(year))) if isinstance(year, (float, str)) and year else str(year)
        return year_str.strip() if year_str.strip().isdigit() else ''
    except:
        return ''

def is_fsrdc_related(work: Dict) -> bool:
    try:
        # Combine all text fields into a single string for one-time search
        text_fields = [
            work.get('title', '').lower(),
            work.get('abstract', '').lower(),
            work.get('fullText', '').lower(),
            ' '.join([kw.lower() for kw in work.get('keywords', []) if kw]),
            ' '.join([inst.get('name', '').lower() for inst in work.get('institutions', []) if inst])
        ]
        combined_text = ' '.join([field for field in text_fields if field])

        # Use regex for faster matching
        return bool(FSRDC_PATTERN.search(combined_text))
    except Exception as e:
        logging.debug(f"FSRDC check error: {str(e)}")
        return False

def infer_rdc(work: Dict) -> str:
    institutions = ' '.join(
        inst.get('name', '').lower() for inst in work.get('institutions', [])
    )

    for keyword, rdc_name in RDC_MAPPING.items():
        if keyword in institutions:
            return rdc_name
    return 'Unknown RDC'

def search_core_by_author_best_match(author_name: str, project_title: str) -> Optional[Dict]:
    if not author_name or author_name.lower() in ['unknown pi', '']:
        return None

    params = {
        'q': f'authors:"{author_name}"',
        'limit': 20,
        'offset': 0
    }
    headers = {
        'Authorization': f'Bearer {CORE_API_KEY}',
        'Accept': 'application/json'
    }

    data = safe_api_call(CORE_API_URL, headers=headers, params=params)
    if not data or 'results' not in data:
        return None

    results = data['results']
    fsrdc_results = [r for r in results if is_fsrdc_related(r)]

    # First try to find an exact match with the project title
    for work in fsrdc_results:
        if project_title.lower() in work.get('title', '').lower():
            return work

    # Otherwise return the first FSRDC-related result
    return fsrdc_results[0] if fsrdc_results else None

def process_pi_record(args):
    pi_data, index = args
    pi_name = str(pi_data.get('author', 'Unknown PI')).strip()
    title = str(pi_data.get('title', 'Untitled Project')).strip()
    year_start = clean_year(pi_data.get('year'))
    year_end = str(int(year_start) + 3) if year_start.isdigit() else ''

    record = {
        'ProjID': f"PROJ_{index:05d}",
        'ProjectStatus': 'Unknown',
        'ProjectTitle': title,
        'ProjectRDC': '',
        'ProjectYearStarted': year_start,
        'ProjectYearEnded': year_end,
        'ProjectPI': pi_name,
        'OutputTitle': '',
        'OutputBiblio': '',
        'OutputType': '',
        'OutputStatus': '',
        'OutputVenue': '',
        'OutputYear': '',
        'OutputMonth': '',
        'OutputVolume': '',
        'OutputNumber': '',
        'OutputPages': '',
        'DOI': '',
        'Source': 'CORE'
    }

    current_year = datetime.now().year
    if year_start.isdigit() and year_end.isdigit():
        s, e = int(year_start), int(year_end)
        if e < current_year:
            record['ProjectStatus'] = 'Completed'
        elif s <= current_year <= e:
            record['ProjectStatus'] = 'Active'
        else:
            record['ProjectStatus'] = 'Planned'

    if pi_name.lower() in ['unknown pi', '']:
        return record, index

    # Add small delay to avoid overwhelming the API
    sleep(REQUEST_DELAY)

    best_work = search_core_by_author_best_match(pi_name, title)
    if not best_work:
        return record, index

    record['ProjectRDC'] = infer_rdc(best_work)
    record['OutputTitle'] = best_work.get('title', '')
    doc_type = best_work.get('documentType', '').lower()
    record['OutputType'] = (
        'Journal Article' if 'journal' in doc_type else
        'Conference Paper' if 'conference' in doc_type else
        'Thesis' if 'thesis' in doc_type else 'Other'
    )
    record['OutputStatus'] = 'Published'
    record['OutputVenue'] = best_work.get('publisher', '')

    pub_date = best_work.get('publishedDate', '')
    if pub_date:
        parts = pub_date.split('-')
        record['OutputYear'] = parts[0] if len(parts) > 0 else ''
        record['OutputMonth'] = parts[1] if len(parts) > 1 else ''

    doi = best_work.get('doi', '')
    record['DOI'] = doi
    record['OutputBiblio'] = f"DOI: {doi}" if doi else ''

    record['OutputVolume'] = best_work.get('volume', '')
    record['OutputNumber'] = best_work.get('issue', '')
    record['OutputPages'] = best_work.get('pages', '')

    return record, index

def print_progress(current: int, total: int, start_time: float, valid_count: int):
    elapsed = time() - start_time
    percent = (current / total) * 100
    records_per_sec = current / elapsed if elapsed > 0 else 0
    eta = (elapsed / current) * (total - current) if current > 0 else 0

    sys.stdout.write(
        f'\rProcessed: {current}/{total} ({percent:.1f}%) | '
        f'Valid: {valid_count} | '
        f'Speed: {records_per_sec:.2f} rec/s | '
        f'ETA: {str(timedelta(seconds=int(eta)))}'
    )
    sys.stdout.flush()

def process_batch(batch, start_idx):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a list of (row, index) tuples to pass to the process_pi_record function
        args_list = [(row, start_idx + i) for i, row in enumerate(batch)]

        # Process records in parallel
        for result, _ in executor.map(process_pi_record, args_list):
            results.append(result)

    return results

def enrich_data(input_csv: str, output_csv: str, batch_size: int = 100):
    try:
        df = pd.read_csv(input_csv, dtype={'year': object, 'author': object, 'title': object})
        df = df.fillna({'author': 'Unknown PI', 'title': 'Untitled Project', 'year': ''})

        total = len(df)
        results = []
        start_time = time()
        partial_output_file = output_csv.replace('.csv', '_partial.csv')

        print(f"🔎 Starting enrichment for {total} records using CORE API with parallelism...\n")
        print(f"Using {MAX_WORKERS} workers and batch size of {batch_size}\n")

        # Process in batches for better memory management and progress reporting
        processed_count = 0
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = df.iloc[batch_start:batch_end].to_dict('records')

            batch_results = process_batch(batch, batch_start + 1)
            results.extend(batch_results)

            processed_count += len(batch)
            valid_count = len([r for r in results if r.get('OutputTitle')])

            print_progress(processed_count, total, start_time, valid_count)

            # Save progress after each batch
            pd.DataFrame(results).to_csv(partial_output_file, index=False)
            print(f"\n💾 Autosaved progress at record {processed_count}")

        output_df = pd.DataFrame(results)
        output_df.to_csv(output_csv, index=False)

        elapsed_time = time() - start_time
        print(f"\n✅ Completed in {elapsed_time:.2f} seconds! Saved {len(output_df)} records to {output_csv}")
        print("\n📊 Summary:")
        print(f"- Projects with PI info: {len(output_df)}")
        print(f"- With RDC info: {len(output_df[output_df['ProjectRDC'] != ''])}")
        print(f"- With Output info: {len(output_df[output_df['OutputTitle'] != ''])}")
        print(f"- With DOI: {len(output_df[output_df['DOI'] != ''])}")
        print(f"- Average processing speed: {total / elapsed_time:.2f} records per second")

    except Exception as e:
        print(f"\n❌ Error in main processing: {str(e)}")

if __name__ == '__main__':
    input_file = 'combined_complete_records.csv'
    output_file = 'enriched_research_outputs_core.csv'

    # Customize these parameters based on your system and API limits
    BATCH_SIZE = 100

    enrich_data(input_file, output_file, BATCH_SIZE)


from time import sleep, time
from datetime import datetime, timedelta
import sys
import logging
from typing import Dict, List, Optional, Union
import concurrent.futures
from functools import lru_cache
import os

# Configure logging
logging.basicConfig(level=logging.CRITICAL)

# Configuration
CORE_API_URL = "https://api.core.ac.uk/v3/search/works"
CORE_API_KEY = "JitbKoV5gnCsEOXGWywqeA9zdNcvZSul"  # Replace with your own if needed
REQUEST_DELAY = 0.1  # Reduced delay between API calls
MAX_RETRIES = 5
RETRY_BACKOFF = [1, 2, 4, 8, 15]
MAX_WORKERS = 10  # Adjust based on your network and API rate limits

# FSRDC keywords (expanded list) - compiled for faster matching
FSRDC_KEYWORDS = [
    "census bureau", "fsrdc", "federal statistical research data center",
    "restricted microdata", "irs", "bea", "confidentiality review",
    "michigan rdc", "texas rdc", "boston rdc", "cornell rdc", "berkeley rdc",
    "annual survey of manufactures", "census of construction industries",
    "census of finance, insurance, and real estate", "ncrdc", "nyc rdc",
    "longitudinal employer-household dynamics", "lehd", "ces", "qwi",
    "confidential data", "restricted data", "statistical disclosure limitation",
    "title 13", "title 26", "cbms", "sbo", "economic census"
]

# Pre-compile regex patterns for faster matching
FSRDC_PATTERN = re.compile('|'.join(map(re.escape, FSRDC_KEYWORDS)))

# RDC mapping for faster lookups
RDC_MAPPING = {
    'michigan': 'Michigan RDC',
    'texas': 'Texas RDC',
    'boston': 'Boston RDC',
    'cornell': 'Cornell RDC',
    'berkeley': 'Berkeley RDC',
    'new york': 'NYC RDC',
    'north carolina': 'NCRDC'
}

# Session object for connection pooling
session = requests.Session()

def safe_api_call(url: str, headers: Optional[Dict] = None, params: Optional[Dict] = None) -> Optional[Dict]:
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, headers=headers, params=params, timeout=30)
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                sleep(retry_after)
                continue
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                logging.debug(f"API call failed: {str(e)}")
                return None
            sleep(RETRY_BACKOFF[attempt])
    return None

@lru_cache(maxsize=1024)
def clean_year(year: Union[str, int, float]) -> str:
    if pd.isna(year):
        return ''
    try:
        year_str = str(int(float(year))) if isinstance(year, (float, str)) and year else str(year)
        return year_str.strip() if year_str.strip().isdigit() else ''
    except:
        return ''

def is_fsrdc_related(work: Dict) -> bool:
    try:
        # Combine all text fields into a single string for one-time search
        text_fields = [
            work.get('title', '').lower(),
            work.get('abstract', '').lower(),
            work.get('fullText', '').lower(),
            ' '.join([kw.lower() for kw in work.get('keywords', []) if kw]),
            ' '.join([inst.get('name', '').lower() for inst in work.get('institutions', []) if inst])
        ]
        combined_text = ' '.join([field for field in text_fields if field])

        # Use regex for faster matching
        return bool(FSRDC_PATTERN.search(combined_text))
    except Exception as e:
        logging.debug(f"FSRDC check error: {str(e)}")
        return False

def infer_rdc(work: Dict) -> str:
    institutions = ' '.join(
        inst.get('name', '').lower() for inst in work.get('institutions', [])
    )

    for keyword, rdc_name in RDC_MAPPING.items():
        if keyword in institutions:
            return rdc_name
    return 'Unknown RDC'

def search_core_by_author_best_match(author_name: str, project_title: str) -> Optional[Dict]:
    if not author_name or author_name.lower() in ['unknown pi', '']:
        return None

    params = {
        'q': f'authors:"{author_name}"',
        'limit': 20,
        'offset': 0
    }
    headers = {
        'Authorization': f'Bearer {CORE_API_KEY}',
        'Accept': 'application/json'
    }

    data = safe_api_call(CORE_API_URL, headers=headers, params=params)
    if not data or 'results' not in data:
        return None

    results = data['results']
    fsrdc_results = [r for r in results if is_fsrdc_related(r)]

    # First try to find an exact match with the project title
    for work in fsrdc_results:
        if project_title.lower() in work.get('title', '').lower():
            return work

    # Otherwise return the first FSRDC-related result
    return fsrdc_results[0] if fsrdc_results else None

def process_pi_record(args):
    pi_data, index = args
    pi_name = str(pi_data.get('author', 'Unknown PI')).strip()
    title = str(pi_data.get('title', 'Untitled Project')).strip()
    year_start = clean_year(pi_data.get('year'))
    year_end = str(int(year_start) + 3) if year_start.isdigit() else ''

    record = {
        'ProjID': f"PROJ_{index:05d}",
        'ProjectStatus': 'Unknown',
        'ProjectTitle': title,
        'ProjectRDC': '',
        'ProjectYearStarted': year_start,
        'ProjectYearEnded': year_end,
        'ProjectPI': pi_name,
        'OutputTitle': '',
        'OutputBiblio': '',
        'OutputType': '',
        'OutputStatus': '',
        'OutputVenue': '',
        'OutputYear': '',
        'OutputMonth': '',
        'OutputVolume': '',
        'OutputNumber': '',
        'OutputPages': '',
        'DOI': '',
        'Source': 'CORE'
    }

    current_year = datetime.now().year
    if year_start.isdigit() and year_end.isdigit():
        s, e = int(year_start), int(year_end)
        if e < current_year:
            record['ProjectStatus'] = 'Completed'
        elif s <= current_year <= e:
            record['ProjectStatus'] = 'Active'
        else:
            record['ProjectStatus'] = 'Planned'

    if pi_name.lower() in ['unknown pi', '']:
        return record, index

    # Add small delay to avoid overwhelming the API
    sleep(REQUEST_DELAY)

    best_work = search_core_by_author_best_match(pi_name, title)
    if not best_work:
        return record, index

    record['ProjectRDC'] = infer_rdc(best_work)
    record['OutputTitle'] = best_work.get('title', '')
    doc_type = best_work.get('documentType', '').lower()
    record['OutputType'] = (
        'Journal Article' if 'journal' in doc_type else
        'Conference Paper' if 'conference' in doc_type else
        'Thesis' if 'thesis' in doc_type else 'Other'
    )
    record['OutputStatus'] = 'Published'
    record['OutputVenue'] = best_work.get('publisher', '')

    pub_date = best_work.get('publishedDate', '')
    if pub_date:
        parts = pub_date.split('-')
        record['OutputYear'] = parts[0] if len(parts) > 0 else ''
        record['OutputMonth'] = parts[1] if len(parts) > 1 else ''

    doi = best_work.get('doi', '')
    record['DOI'] = doi
    record['OutputBiblio'] = f"DOI: {doi}" if doi else ''

    record['OutputVolume'] = best_work.get('volume', '')
    record['OutputNumber'] = best_work.get('issue', '')
    record['OutputPages'] = best_work.get('pages', '')

    return record, index

def print_progress(current: int, total: int, start_time: float, valid_count: int, already_processed: int):
    elapsed = time() - start_time
    percent = ((current + already_processed) / total) * 100
    records_per_sec = current / elapsed if elapsed > 0 else 0
    eta = (elapsed / current) * (total - current - already_processed) if current > 0 else 0

    sys.stdout.write(
        f'\rProcessed: {current + already_processed}/{total} ({percent:.1f}%) | '
        f'Valid: {valid_count} | '
        f'Speed: {records_per_sec:.2f} rec/s | '
        f'ETA: {str(timedelta(seconds=int(eta)))}'
    )
    sys.stdout.flush()

def process_batch(batch, start_idx):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a list of (row, index) tuples to pass to the process_pi_record function
        args_list = [(row, start_idx + i) for i, row in enumerate(batch)]

        # Process records in parallel
        for result, _ in executor.map(process_pi_record, args_list):
            results.append(result)

    return results

def resume_enrichment(input_csv: str, output_csv: str, batch_size: int = 100):
    try:
        # Load the input data
        df_input = pd.read_csv(input_csv, dtype={'year': object, 'author': object, 'title': object})
        df_input = df_input.fillna({'author': 'Unknown PI', 'title': 'Untitled Project', 'year': ''})

        # Check if output file exists, and load it if it does
        if os.path.exists(output_csv):
            df_output = pd.read_csv(output_csv)
            print(f"Found existing output file with {len(df_output)} records")

            # Extract the last processed index from the ProjID column
            if len(df_output) > 0:
                last_proj_id = df_output['ProjID'].iloc[-1]
                last_index = int(last_proj_id.split('_')[1])
                print(f"Last processed record index: {last_index}")
            else:
                last_index = 0

            # We'll continue from the next record
            start_index = last_index + 1
            if start_index > len(df_input):
                print("All records have already been processed!")
                return

            # Get the remaining records to process
            remaining_df = df_input.iloc[start_index-1:]
            total_remaining = len(remaining_df)
            print(f"Continuing with {total_remaining} remaining records (starting from index {start_index})")

        else:
            # No existing output file, start from the beginning
            df_output = pd.DataFrame()
            start_index = 1
            remaining_df = df_input
            total_remaining = len(remaining_df)
            print(f"No existing output file found. Processing all {total_remaining} records")

        # Get the total number of records for progress reporting
        total = len(df_input)
        already_processed = start_index - 1

        # Process the remaining records
        results = []
        start_time = time()
        partial_output_file = output_csv.replace('.csv', '_partial.csv')

        print(f"🔎 Starting enrichment for {total_remaining} records using CORE API with parallelism...\n")
        print(f"Using {MAX_WORKERS} workers and batch size of {batch_size}\n")

        # Process in batches for better memory management and progress reporting
        processed_count = 0
        for batch_start in range(0, total_remaining, batch_size):
            batch_end = min(batch_start + batch_size, total_remaining)
            batch = remaining_df.iloc[batch_start:batch_end].to_dict('records')

            batch_results = process_batch(batch, start_index + batch_start)
            results.extend(batch_results)

            processed_count += len(batch)
            valid_count = len([r for r in results if r.get('OutputTitle')])

            print_progress(processed_count, total, start_time, valid_count, already_processed)

            # Save progress after each batch - append to existing file if it exists
            batch_df = pd.DataFrame(batch_results)

            # On the first batch, we'll either create a new file or append to the existing one
            if batch_start == 0:
                if os.path.exists(output_csv):
                    # Append to existing file
                    batch_df.to_csv(output_csv, mode='a', header=False, index=False)
                else:
                    # Create new file
                    batch_df.to_csv(output_csv, index=False)
            else:
                # Append to the file for subsequent batches
                batch_df.to_csv(output_csv, mode='a', header=False, index=False)

            # Also save to partial output file for backup
            if len(results) > 0:
                pd.DataFrame(results).to_csv(partial_output_file, index=False)

            print(f"\n💾 Saved progress through record {already_processed + processed_count}")

        elapsed_time = time() - start_time
        print(f"\n✅ Completed in {elapsed_time:.2f} seconds! Saved {already_processed + processed_count} records to {output_csv}")
        print("\n📊 Summary:")

        # Get final stats by reading the complete output file
        final_df = pd.read_csv(output_csv)
        print(f"- Total records: {len(final_df)}")
        print(f"- With RDC info: {len(final_df[final_df['ProjectRDC'] != ''])}")
        print(f"- With Output info: {len(final_df[final_df['OutputTitle'] != ''])}")
        print(f"- With DOI: {len(final_df[final_df['DOI'] != ''])}")
        print(f"- Average processing speed: {processed_count / elapsed_time:.2f} records per second")

    except Exception as e:
        print(f"\n❌ Error in main processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    input_file = 'combined_complete_records.csv'
    output_file = 'enriched_research_outputs_core.csv'

    # Customize these parameters based on your system and API limits
    BATCH_SIZE = 100

    resume_enrichment(input_file, output_file, BATCH_SIZE)



# Load your enriched data
file_path = 'enriched_research_outputs_core.csv'
df = pd.read_csv(file_path)

# Check if there's a Source column that might indicate valid matches
if 'Source' in df.columns:
    print("Sources in data:")
    print(df['Source'].value_counts())

# Check if there are any indicators of valid records
cols = df.columns
print("\nColumns in data:", cols)

# Let's create a custom definition of "valid" based on non-empty key fields
valid_df = df.copy()
for col in ['OutputTitle', 'DOI', 'OutputVenue', 'OutputYear']:
    if col in valid_df.columns:
        valid_df = valid_df[valid_df[col].notna() & (valid_df[col].astype(str).str.strip() != '')]
        print(f"After filtering for non-empty {col}: {len(valid_df)} records")

# Save the filtered data
if len(valid_df) < len(df):
    valid_df.to_csv('valid_custom_filter.csv', index=False)
    print(f"\nSaved {len(valid_df)} custom filtered records")
else:
    print("\nNo records were filtered out. Need to investigate further.")