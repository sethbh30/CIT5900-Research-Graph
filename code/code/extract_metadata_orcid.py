import pandas as pd
import requests
import re
from time import sleep, time
from datetime import datetime, timedelta
import logging
import sys

# Disable all logging
logging.basicConfig(level=logging.CRITICAL)

def safe_api_call(url, headers=None, params=None):
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except:
        return None

def process_pi_record(pi_data, index):
    try:
        pi_name = pi_data['author']
        title = pi_data['title']
        year = pi_data['year']

        # ORCID search
        orcid_result = safe_api_call(
            "https://pub.orcid.org/v3.0/search/",
            headers={'Accept': 'application/json'},
            params={'q': f'given-names:{pi_name.split()[0]} AND family-name:{pi_name.split()[-1]}'}
        )

        if not orcid_result or not orcid_result.get('result'):
            return None

        orcid_id = orcid_result['result'][0]['orcid-identifier']['path']
        sleep(1.2)

        # Get ORCID record
        record = safe_api_call(f"https://pub.orcid.org/v3.0/{orcid_id}/record",
                             headers={'Accept': 'application/json'})

        if not record:
            return None

        # Process works
        works = record.get('activities-summary', {}).get('works', {}).get('group', [])
        for group in works:
            for work_summary in group.get('work-summary', []):
                work = safe_api_call(f"https://pub.orcid.org/v3.0/{orcid_id}/work/{work_summary.get('put-code')}",
                                   headers={'Accept': 'application/json'})
                if work and work.get('title'):
                    return {
                        'ProjID': f"PROJ_{index:05d}",
                        'ProjectPI': pi_name,
                        'ProjectTitle': title,
                        'OutputTitle': work['title'].get('title', {}).get('value', ''),
                        'DOI': next((id['external-id-value'] for id in work.get('external-ids', {}).get('external-id', [])
                                    if id.get('external-id-type') == 'doi'), ''),
                        'OutputYear': work.get('publication-date', {}).get('year', {}).get('value', '')
                    }
        return None
    except:
        return None

def print_progress(current, total, start_time, valid_count):
    elapsed = time() - start_time
    percent = (current / total) * 100
    eta = (elapsed / current) * (total - current) if current > 0 else 0

    sys.stdout.write(
        f'\rProcessed: {current}/{total} ({percent:.1f}%) | '
        f'Valid: {valid_count} | '
        f'ETA: {str(timedelta(seconds=int(eta)))}'
    )
    sys.stdout.flush()

def enrich_data(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    total = len(df)
    results = []
    start_time = time()

    print(f"Starting ORCID enrichment for {total} PIs...\n")

    for idx, row in df.iterrows():
        result = process_pi_record(row, idx + 1)
        if result:
            results.append(result)

        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            print_progress(idx + 1, total, start_time, len(results))

    if results:
        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"\n\nCompleted! Saved {len(results)} enriched records to {output_csv}")
    else:
        print("\n\nNo valid records found")

if __name__ == '__main__':
    enrich_data('combined_complete_records.csv', 'enriched_research_outputs.csv')


    import pandas as pd
import requests
import re
from time import sleep, time
from datetime import datetime, timedelta
import logging
import sys
import json

# Configure logging to show only critical errors
logging.basicConfig(level=logging.CRITICAL)

def safe_api_call(url, headers=None, params=None):
    """Make API calls with error handling"""
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None

def extract_doi_info(doi):
    """Extract additional metadata from DOI using CrossRef API"""
    if not doi:
        return {}

    try:
        response = requests.get(f"https://api.crossref.org/works/{doi}", timeout=10)
        if response.status_code == 200:
            data = response.json().get('message', {})
            return {
                'OutputVenue': data.get('container-title', [''])[0],
                'OutputVolume': data.get('volume', ''),
                'OutputNumber': data.get('issue', ''),
                'OutputPage': f"{data.get('page', '')}",
                'OutputMonth': data.get('published-print', {}).get('date-parts', [[0, 0]])[0][1] if len(data.get('published-print', {}).get('date-parts', [[0]])[0]) > 1 else '',
                'OutputBiblio': format_biblio(data)
            }
    except:
        pass
    return {}

def format_biblio(data):
    """Format bibliographic information from CrossRef data"""
    try:
        authors = ", ".join([f"{a.get('family', '')}, {a.get('given', '')}" for a in data.get('author', [])])
        title = data.get('title', [''])[0]
        journal = data.get('container-title', [''])[0]
        volume = data.get('volume', '')
        issue = data.get('issue', '')
        page = data.get('page', '')
        year = data.get('published', {}).get('date-parts', [[0]])[0][0] if 'published' in data else ''

        biblio = f"{authors}. {title}."
        if journal:
            biblio += f" {journal}"
            if volume:
                biblio += f" {volume}"
                if issue:
                    biblio += f"({issue})"
            if page:
                biblio += f", {page}"
        if year:
            biblio += f" ({year})"
        return biblio
    except:
        return ""

def determine_output_type(work):
    """Determine output type from ORCID work data"""
    if not work:
        return "Unknown"

    type_map = {
        'journal-article': 'Journal Article',
        'book': 'Book',
        'book-chapter': 'Book Chapter',
        'conference-paper': 'Conference Paper',
        'dataset': 'Dataset',
        'report': 'Report',
        'patent': 'Patent',
        'dissertation': 'Dissertation',
        'review': 'Review',
        'preprint': 'Preprint'
    }

    work_type = work.get('type', '').lower()
    return type_map.get(work_type, work_type.replace('-', ' ').title() if work_type else 'Publication')

def process_pi_record(pi_data, index):
    """Process PI record and enrich with ORCID data"""
    try:
        if not pi_data:
            return None

        # Try different column names that might contain PI name
        pi_name = None
        for key in ['author', 'ProjectPI', 'PI', 'project_pi', 'pi_name', 'name']:
            if key in pi_data and pi_data[key]:
                pi_name = pi_data[key]
                break

        # If PI name still not found, log and return
        if not pi_name or not isinstance(pi_name, str) or len(pi_name.split()) < 2:
            print(f"Debug - Missing or invalid PI name for record {index}. Keys: {list(pi_data.keys())}")
            return None

        # Try different column names for project title
        project_title = None
        for key in ['title', 'ProjectTitle', 'project_title']:
            if key in pi_data and pi_data[key]:
                project_title = pi_data[key]
                break

        project_year = pi_data.get('year', '')

        # Generate a project ID - either use existing or create new
        proj_id = pi_data.get('ProjID', '') or f"PROJ_{index:05d}"

        # Debug print
        print(f"Debug - Processing PI: {pi_name}, Title: {project_title[:30]}...")

        # ORCID search - try different name formats
        orcid_result = None
        name_parts = pi_name.split()

        # Try first with standard format
        search_query = f'given-names:{name_parts[0]} AND family-name:{name_parts[-1]}'
        orcid_result = safe_api_call(
            "https://pub.orcid.org/v3.0/search/",
            headers={'Accept': 'application/json'},
            params={'q': search_query}
        )

        # If that fails, try with just the last name
        if not orcid_result or not orcid_result.get('result'):
            search_query = f'family-name:{name_parts[-1]}'
            orcid_result = safe_api_call(
                "https://pub.orcid.org/v3.0/search/",
                headers={'Accept': 'application/json'},
                params={'q': search_query}
            )

        if not orcid_result or not orcid_result.get('result'):
            print(f"Debug - No ORCID found for {pi_name}")
            return None

        orcid_id = orcid_result['result'][0]['orcid-identifier']['path']
        sleep(1.2)  # Respect rate limits

        # Get ORCID record
        record = safe_api_call(f"https://pub.orcid.org/v3.0/{orcid_id}/record",
                             headers={'Accept': 'application/json'})

        if not record:
            return None

        # Extract employment history for project dates
        employments = record.get('activities-summary', {}).get('employments', {}).get('employment-summary', [])
        project_start = next((emp.get('start-date', {}).get('year', {}).get('value') for emp in employments if emp.get('start-date')), '')
        project_end = next((emp.get('end-date', {}).get('year', {}).get('value') for emp in employments if emp.get('end-date')), 'Ongoing')

        # Extract organization/institution (RDC)
        project_rdc = next((emp.get('organization', {}).get('name') for emp in employments), '')

        # Process works to find outputs
        works = record.get('activities-summary', {}).get('works', {}).get('group', [])
        results = []

        for group in works[:min(5, len(works))]:  # Limit to max 5 works per PI
            for work_summary in group.get('work-summary', [])[:1]:  # Take first work from each group
                work = safe_api_call(f"https://pub.orcid.org/v3.0/{orcid_id}/work/{work_summary.get('put-code')}",
                                  headers={'Accept': 'application/json'})

                if work and work.get('title'):
                    # Extract DOI if available
                    doi = next((id['external-id-value'] for id in work.get('external-ids', {}).get('external-id', [])
                              if id.get('external-id-type') == 'doi'), '')

                    # Get publication year and month
                    pub_date = work.get('publication-date', {})
                    year = pub_date.get('year', {}).get('value', '')
                    month = pub_date.get('month', {}).get('value', '')

                    # Get additional metadata from DOI
                    doi_metadata = extract_doi_info(doi) if doi else {}

                    # Determine output type and status
                    output_type = determine_output_type(work)
                    output_status = "Published" if year else "In Progress"

                    result = {
                        'ProjID': proj_id,
                        'ProjectStatus': "Completed" if project_end and project_end != 'Ongoing' else "Active",
                        'ProjectTitle': project_title,
                        'ProjectRDC': project_rdc,
                        'ProjectYearStarted': project_start or '',
                        'ProjectYearEnded': project_end or '',
                        'ProjectPI': pi_name,
                        'OutputTitle': work['title'].get('title', {}).get('value', ''),
                        'OutputBiblio': doi_metadata.get('OutputBiblio', ''),
                        'OutputType': output_type,
                        'OutputStatus': output_status,
                        'OutputVenue': doi_metadata.get('OutputVenue', ''),
                        'OutputYear': year,
                        'OutputMonth': doi_metadata.get('OutputMonth', month),
                        'OutputVolume': doi_metadata.get('OutputVolume', ''),
                        'OutputNumber': doi_metadata.get('OutputNumber', ''),
                        'OutputPage': doi_metadata.get('OutputPage', ''),
                        'DOI': doi
                    }
                    results.append(result)

        return results[0] if results else None
    except Exception as e:
        print(f"Error processing record: {e}")
        return None

def print_progress(current, total, start_time, valid_count):
    """Print progress information"""
    elapsed = time() - start_time
    percent = (current / total) * 100
    eta = (elapsed / current) * (total - current) if current > 0 else 0

    sys.stdout.write(
        f'\rProcessed: {current}/{total} ({percent:.1f}%) | '
        f'Valid: {valid_count} | '
        f'ETA: {str(timedelta(seconds=int(eta)))}'
    )
    sys.stdout.flush()

def enrich_data(input_csv, output_csv):
    """Main function to enrich data from CSV using ORCID API"""
    try:
        # Read input data and print column names to debug
        df = pd.read_csv(input_csv)
        print(f"Input CSV columns: {df.columns.tolist()}")
        print(f"First row sample: {df.iloc[0].to_dict()}")

        # If the CSV is using the enriched columns already, preserve data
        output_data = []
        if 'ProjectPI' in df.columns and 'ProjectTitle' in df.columns:
            # The file already has some structure we should preserve
            print("Using existing ProjectPI and ProjectTitle columns...")
            # Create a copy to avoid modifying the original
            output_df = df.copy()
        else:
            # We need to create a new dataframe
            output_df = pd.DataFrame()

        total = len(df)
        results = []
        start_time = time()

        print(f"Starting ORCID enrichment for {total} PIs...\n")

        # Process a small sample first to debug
        sample_size = min(5, total)
        print(f"Processing sample of {sample_size} records first...")

        for idx, row in df.head(sample_size).iterrows():
            result = process_pi_record(row.to_dict(), idx + 1)
            if result:
                results.append(result)
                print(f"Success! Found ORCID data for {row.get('ProjectPI', row.get('author', 'Unknown'))}")
            else:
                print(f"Failed to find ORCID data for {row.get('ProjectPI', row.get('author', 'Unknown'))}")

        # If sample worked, process the rest
        if results:
            print(f"Sample processing successful. Processing remaining records...")
            for idx, row in df.iloc[sample_size:].iterrows():
                result = process_pi_record(row.to_dict(), idx + 1)
                if result:
                    results.append(result)

                if (idx + 1) % 10 == 0 or (idx + 1) == total:
                    print_progress(idx + 1, total, start_time, len(results))
        else:
            # If no results from sample, try to generate placeholder data
            print("No results from ORCID API. Generating placeholder data from input file...")
            for idx, row in df.iterrows():
                row_dict = row.to_dict()

                # Create basic record from existing data
                result = {
                    'ProjID': row_dict.get('ProjID', f"PROJ_{idx+1:05d}"),
                    'ProjectStatus': row_dict.get('ProjectStatus', 'Unknown'),
                    'ProjectTitle': row_dict.get('ProjectTitle', row_dict.get('title', 'Unknown')),
                    'ProjectRDC': row_dict.get('ProjectRDC', 'Unknown'),
                    'ProjectYearStarted': row_dict.get('ProjectYearStarted', ''),
                    'ProjectYearEnded': row_dict.get('ProjectYearEnded', ''),
                    'ProjectPI': row_dict.get('ProjectPI', row_dict.get('author', 'Unknown')),
                    'OutputTitle': row_dict.get('OutputTitle', row_dict.get('title', 'Unknown')),
                    'OutputBiblio': row_dict.get('OutputBiblio', ''),
                    'OutputType': row_dict.get('OutputType', 'Publication'),
                    'OutputStatus': row_dict.get('OutputStatus', 'Unknown'),
                    'OutputVenue': row_dict.get('OutputVenue', ''),
                    'OutputYear': row_dict.get('OutputYear', row_dict.get('year', '')),
                    'OutputMonth': row_dict.get('OutputMonth', ''),
                    'OutputVolume': row_dict.get('OutputVolume', ''),
                    'OutputNumber': row_dict.get('OutputNumber', ''),
                    'OutputPage': row_dict.get('OutputPage', ''),
                    'DOI': row_dict.get('DOI', '')
                }
                results.append(result)

                if (idx + 1) % 100 == 0 or (idx + 1) == total:
                    print_progress(idx + 1, total, start_time, len(results))

        if results:
            output_df = pd.DataFrame(results)

            # Ensure all required columns are present
            required_columns = [
                'ProjID', 'ProjectStatus', 'ProjectTitle', 'ProjectRDC', 'ProjectYearStarted',
                'ProjectYearEnded', 'ProjectPI', 'OutputTitle', 'OutputBiblio', 'OutputType',
                'OutputStatus', 'OutputVenue', 'OutputYear', 'OutputMonth', 'OutputVolume',
                'OutputNumber', 'OutputPage', 'DOI'
            ]

            for col in required_columns:
                if col not in output_df.columns:
                    output_df[col] = ''

            # Reorder columns to match required order
            output_df = output_df[required_columns]

            # Save to CSV
            output_df.to_csv(output_csv, index=False)
            print(f"\n\nCompleted! Saved {len(results)} enriched records to {output_csv}")
        else:
            print("\n\nNo valid records found")
    except Exception as e:
        print(f"Error in data enrichment: {e}")
        import traceback
        traceback.print_exc()

def main():
    # First try to read the existing enriched file to preserve data
    try:
        print("Checking if enriched_research_outputs.csv exists...")
        existing_df = pd.read_csv('enriched_research_outputs.csv')
        print(f"Found existing file with {len(existing_df)} records and columns: {existing_df.columns.tolist()}")

        print("Would you like to use the existing enriched file as input? (y/n)")
        use_existing = input().lower().strip() == 'y'

        if use_existing:
            input_file = 'enriched_research_outputs.csv'
            output_file = 'enriched_research_outputs_updated.csv'
        else:
            input_file = 'combined_complete_records.csv'
            output_file = 'enriched_research_outputs.csv'
    except:
        print("No existing enriched file found or error reading it.")
        input_file = 'combined_complete_records.csv'
        output_file = 'enriched_research_outputs.csv'

    # Ask user to confirm the column mapping
    print(f"\nReading from: {input_file}")
    print(f"Writing to: {output_file}")
    print("\nPress Enter to continue, or Ctrl+C to cancel...")
    input()

    enrich_data(input_file, output_file)

if __name__ == '__main__':
    main()

import pandas as pd
import requests
import time
import re

def get_orcid_works(orcid_id):
    headers = {"Accept": "application/json"}
    url = f"https://pub.orcid.org/v3.0/{orcid_id}/works"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json().get("group", [])
    return []

def get_work_details(orcid_id, put_code):
    headers = {"Accept": "application/json"}
    url = f"https://pub.orcid.org/v3.0/{orcid_id}/work/{put_code}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    return {}

def match_doi(work, target_doi):
    for ext_id in work.get("external-ids", {}).get("external-id", []):
        if ext_id.get("external-id-type", "").lower() == "doi":
            doi = ext_id.get("external-id-value", "").lower()
            if doi == target_doi.lower():
                return True
    return False

def extract_volume_issue_page(citation):
    match = re.search(r'(\d+)\((\d+)\),\s*(\d+-\d+)', citation)
    if match:
        return match.groups()
    return '', '', ''

def enrich_missing_fields(row):
    if pd.notna(row['OutputVenue']) and pd.notna(row['OutputVolume']):
        return row  # Already enriched

    orcid_id = row['ORCID ID']
    doi = row['DOI']
    if pd.isna(orcid_id) or pd.isna(doi):
        return row

    try:
        works = get_orcid_works(orcid_id)
        for group in works:
            for summary in group.get("work-summary", []):
                put_code = summary.get("put-code")
                if not put_code:
                    continue

                work = get_work_details(orcid_id, put_code)
                if match_doi(work, doi):
                    row['OutputVenue'] = work.get('journal-title', {}).get('value', row['OutputVenue'])

                    citation = work.get('citation', {}).get('citation-value', '')
                    volume, number, page = extract_volume_issue_page(citation)
                    row['OutputVolume'] = volume or row['OutputVolume']
                    row['OutputNumber'] = number or row['OutputNumber']
                    row['OutputPage'] = page or row['OutputPage']
                    return row
                time.sleep(0.1)
    except Exception as e:
        print(f"Error processing {orcid_id}: {e}")
    return row

def main():
    df_outputs = pd.read_csv('enriched_research_outputs_updated.csv')
    df_orcids = pd.read_csv('pi_orcid_ids.csv')

    # Merge ORCID IDs into the output dataset
    df = df_outputs.merge(df_orcids, left_on='ProjectPI', right_on='PI Name', how='left')
    df['ORCID ID'] = df['ORCID ID'].fillna('')

    print("Enriching records with missing metadata from ORCID...")
    df = df.apply(enrich_missing_fields, axis=1)

    df.to_csv('enriched_research_outputs_full.csv', index=False)
    print("Saved enriched results to enriched_research_outputs_full.csv")

if __name__ == "__main__":
    main()
import pandas as pd
import requests
import time
import re

def get_work_details_by_doi(orcid_id, target_doi):
    headers = {"Accept": "application/json"}
    base_url = f"https://pub.orcid.org/v3.0/{orcid_id}/works"
    response = requests.get(base_url, headers=headers)
    if response.status_code != 200:
        return {}

    works = response.json().get("group", [])
    for group in works:
        for summary in group.get("work-summary", []):
            put_code = summary.get("put-code")
            ext_ids = summary.get("external-ids", {}).get("external-id", [])
            for ext in ext_ids:
                if ext.get("external-id-type", "").lower() == "doi" and ext.get("external-id-value", "").lower() == target_doi.lower():
                    # Found matching work; fetch full metadata
                    detail_url = f"https://pub.orcid.org/v3.0/{orcid_id}/work/{put_code}"
                    detail_response = requests.get(detail_url, headers=headers)
                    if detail_response.status_code == 200:
                        return detail_response.json()
    return {}

def extract_metadata_from_citation(citation_text):
    # Example: "Environmental Science & Technology 2023, 57 (15), 5746-5753"
    volume, number, page = '', '', ''
    match = re.search(r'(\d{4})[,\s]*(\d+)\s*\((\d+)\)\s*,\s*([\d\-]+)', citation_text)
    if match:
        _, volume, number, page = match.groups()
    return volume, number, page

def enrich_row(row):
    if pd.isna(row['ORCID ID']) or pd.isna(row['DOI']):
        return row

    orcid_id = row['ORCID ID'].strip()
    doi = row['DOI'].strip()

    try:
        work = get_work_details_by_doi(orcid_id, doi)
        if not work:
            return row

        # Month
        month = work.get("publication-date", {}).get("month", {}).get("value")
        if pd.isna(row['OutputMonth']) and month:
            row['OutputMonth'] = month

        # Citation
        citation = work.get("citation", {}).get("citation-value", "")
        if citation:
            vol, num, page = extract_metadata_from_citation(citation)
            if pd.isna(row['OutputVolume']) and vol:
                row['OutputVolume'] = vol
            if pd.isna(row['OutputNumber']) and num:
                row['OutputNumber'] = num
            if pd.isna(row['OutputPage']) and page:
                row['OutputPage'] = page

    except Exception as e:
        print(f"Error processing ORCID {orcid_id}: {e}")

    time.sleep(0.2)  # Respect ORCID rate limits
    return row

def main():
    df = pd.read_csv("enriched_research_outputs_full.csv")

    # Only process rows where these fields are missing
    mask = df['OutputVolume'].isna() | df['OutputMonth'].isna() | df['OutputNumber'].isna() | df['OutputPage'].isna()
    df.loc[mask] = df.loc[mask].apply(enrich_row, axis=1)

    df.to_csv("enriched_research_outputs_orcid_enriched.csv", index=False)
    print("Metadata enrichment complete.")

if __name__ == "__main__":
    main()

import pandas as pd
import requests
import re
from time import sleep, time
from datetime import datetime, timedelta
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.CRITICAL)

def safe_api_call(url, headers=None, params=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                logging.debug(f"API call failed after {max_retries} attempts: {str(e)}")
                return None
            sleep(1)
    return None

def clean_year(year):
    """Convert year to string and clean it"""
    if pd.isna(year):
        return ''
    try:
        year_str = str(int(float(year))) if isinstance(year, (float, str)) and year else str(year)
        return year_str.strip() if year_str.strip().isdigit() else ''
    except:
        return ''

def process_pi_record(pi_data, index):
    try:
        # Extract basic PI information with defaults
        pi_name = str(pi_data.get('author', 'Unknown PI')).strip()
        title = str(pi_data.get('title', 'Untitled Project')).strip()
        year_start = clean_year(pi_data.get('year'))

        # Calculate end year (3 years after start if available)
        year_end = ''
        if year_start and year_start.isdigit():
            year_end = str(int(year_start) + 3)

        # Generate basic project info
        project_info = {
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
            'OutputPage': ''
        }

        # Try to determine project status based on dates
        current_year = datetime.now().year
        if year_start and year_end and year_start.isdigit() and year_end.isdigit():
            start_year = int(year_start)
            end_year = int(year_end)
            if end_year < current_year:
                project_info['ProjectStatus'] = 'Completed'
            elif start_year <= current_year <= end_year:
                project_info['ProjectStatus'] = 'Active'
            else:
                project_info['ProjectStatus'] = 'Planned'

        # Skip ORCID lookup if PI name is missing or default
        if pi_name.lower() in ['unknown pi', '']:
            return project_info

        # ORCID search
        orcid_result = safe_api_call(
            "https://pub.orcid.org/v3.0/search/",
            headers={'Accept': 'application/json'},
            params={'q': f'given-names:{pi_name.split()[0]} AND family-name:{pi_name.split()[-1]}'}
        )

        if not orcid_result or not orcid_result.get('result'):
            return project_info

        orcid_id = orcid_result['result'][0]['orcid-identifier']['path']
        sleep(1.2)  # Respect ORCID API rate limits

        # Get ORCID record
        record = safe_api_call(f"https://pub.orcid.org/v3.0/{orcid_id}/record",
                             headers={'Accept': 'application/json'})

        if not record:
            return project_info

        # Try to get employment info for RDC
        employments = record.get('activities-summary', {}).get('employments', {}).get('employment-summary', [])
        if employments:
            # Filter out employments without end dates
            valid_employments = [emp for emp in employments
                               if emp.get('end-date', {}).get('year', {}).get('value')]

            if valid_employments:
                latest_employment = max(
                    valid_employments,
                    key=lambda x: int(x.get('end-date', {}).get('year', {}).get('value', '0'))
                )
                project_info['ProjectRDC'] = latest_employment.get('organization', {}).get('name', '')

        # Process works to find most relevant publication
        works = record.get('activities-summary', {}).get('works', {}).get('group', [])
        most_relevant_work = None

        for group in works:
            for work_summary in group.get('work-summary', []):
                work = safe_api_call(f"https://pub.orcid.org/v3.0/{orcid_id}/work/{work_summary.get('put-code')}",
                                   headers={'Accept': 'application/json'})
                if work:
                    # Prefer works that match the project title or are recent
                    work_title = work['title'].get('title', {}).get('value', '').lower()
                    if title.lower() in work_title or not most_relevant_work:
                        most_relevant_work = work

        if most_relevant_work:
            # Extract publication details
            project_info['OutputTitle'] = most_relevant_work['title'].get('title', {}).get('value', '')

            # Get publication type
            project_info['OutputType'] = most_relevant_work.get('type', '')

            # Get publication status
            project_info['OutputStatus'] = 'Published' if most_relevant_work.get('publication-date') else 'Unpublished'

            # Get journal or venue information
            journal_title = most_relevant_work.get('journal-title', {}).get('value', '')
            if not journal_title:
                # Try to get from external IDs
                for ext_id in most_relevant_work.get('external-ids', {}).get('external-id', []):
                    if ext_id.get('external-id-type') == 'issn':
                        journal_title = ext_id.get('external-id-value', '')
                        break
            project_info['OutputVenue'] = journal_title

            # Get publication date
            pub_date = most_relevant_work.get('publication-date', {})
            if pub_date:
                project_info['OutputYear'] = pub_date.get('year', {}).get('value', '')
                project_info['OutputMonth'] = pub_date.get('month', {}).get('value', '')

            # Get bibliographic info
            doi = next((id['external-id-value'] for id in most_relevant_work.get('external-ids', {}).get('external-id', [])
                      if id.get('external-id-type') == 'doi'), '')
            if doi:
                project_info['OutputBiblio'] = f"doi:{doi}"

            # Try to get volume, issue, pages from citation
            citation = most_relevant_work.get('citation', {})
            if citation and citation.get('citation-type') == 'formatted-unspecified':
                citation_text = citation.get('citation-value', '')
                # Simple pattern matching for common citation elements
                vol_match = re.search(r'vol(?:ume)?\.?\s*(\d+)', citation_text, re.I)
                if vol_match:
                    project_info['OutputVolume'] = vol_match.group(1)

                issue_match = re.search(r'(no|issue|nr)\.?\s*(\d+)', citation_text, re.I)
                if issue_match:
                    project_info['OutputNumber'] = issue_match.group(2)

                pages_match = re.search(r'p(?:p|ages)?\.?\s*(\d+\s*[-–]\s*\d+|\d+)', citation_text, re.I)
                if pages_match:
                    project_info['OutputPage'] = pages_match.group(1)

        return project_info

    except Exception as e:
        logging.debug(f"Error processing record {index}: {str(e)}")
        # Return at least the basic information we have
        return {
            'ProjID': f"PROJ_{index:05d}",
            'ProjectStatus': 'Unknown',
            'ProjectTitle': str(pi_data.get('title', 'Untitled Project')).strip(),
            'ProjectRDC': '',
            'ProjectYearStarted': clean_year(pi_data.get('year')),
            'ProjectYearEnded': str(int(clean_year(pi_data.get('year'))) + 3) if clean_year(pi_data.get('year')).isdigit() else '',
            'ProjectPI': str(pi_data.get('author', 'Unknown PI')).strip(),
            'OutputTitle': '',
            'OutputBiblio': '',
            'OutputType': '',
            'OutputStatus': '',
            'OutputVenue': '',
            'OutputYear': '',
            'OutputMonth': '',
            'OutputVolume': '',
            'OutputNumber': '',
            'OutputPage': ''
        }

def print_progress(current, total, start_time, valid_count):
    elapsed = time() - start_time
    percent = (current / total) * 100
    eta = (elapsed / current) * (total - current) if current > 0 else 0

    sys.stdout.write(
        f'\rProcessed: {current}/{total} ({percent:.1f}%) | '
        f'Valid: {valid_count} | '
        f'ETA: {str(timedelta(seconds=int(eta)))}'
    )
    sys.stdout.flush()

def enrich_data(input_csv, output_csv):
    try:
        # Read CSV with proper handling of mixed types
        df = pd.read_csv(input_csv, dtype={'year': object, 'author': object, 'title': object})

        # Clean the dataframe by filling NA values
        df = df.fillna({'author': 'Unknown PI', 'title': 'Untitled Project', 'year': ''})

        total = len(df)
        results = []
        start_time = time()

        print(f"Starting enrichment for {total} records...\n")

        for idx, row in df.iterrows():
            result = process_pi_record(row, idx + 1)
            results.append(result)

            if (idx + 1) % 10 == 0 or (idx + 1) == total:
                print_progress(idx + 1, total, start_time, len([r for r in results if r.get('OutputTitle')]))

        # Create DataFrame and save
        output_df = pd.DataFrame(results)
        output_df.to_csv(output_csv, index=False)

        print(f"\n\nCompleted! Saved {len(output_df)} records to {output_csv}")
        print("\nSummary of collected data:")
        print(f"- Projects with PI info: {len(output_df)}")
        print(f"- Projects with RDC info: {len(output_df[output_df['ProjectRDC'] != ''])}")
        print(f"- Projects with output info: {len(output_df[output_df['OutputTitle'] != ''])}")

    except Exception as e:
        print(f"\nError in main processing: {str(e)}")
        return

if __name__ == '__main__':
    input_file = 'combined_complete_records.csv'
    output_file = 'enriched_research_outputs_full.csv'
    enrich_data(input_file, output_file)   

import pandas as pd

# Load the full and filtered datasets
full_df = pd.read_csv("enriched_research_outputs_full.csv")
onecode_df = pd.read_csv("enriched_research_outputs_full_onecode.csv")

# Merge using common identifiers (ProjID and OutputTitle are good keys if available)
merged = pd.merge(
    full_df,
    onecode_df[["ProjID", "OutputTitle", "ProjectYearStarted", "ProjectYearEnded", "OutputMonth"]],
    on=["ProjID", "OutputTitle"],
    how="left",
    suffixes=('', '_onecode')
)

# Fill missing values in the full_df with corresponding ones from onecode_df
for col in ["ProjectYearStarted", "ProjectYearEnded", "OutputMonth"]:
    merged[col] = merged[col].combine_first(merged[f"{col}_onecode"])

# Drop the temporary merge columns
merged.drop(columns=[f"{col}_onecode" for col in ["ProjectYearStarted", "ProjectYearEnded", "OutputMonth"]], inplace=True)

# Save the updated CSV
merged.to_csv("enriched_research_outputs_full_updated.csv", index=False)
print("Updated file saved as 'enriched_research_outputs_full_updated.csv'")


import pandas as pd
import re

# Load the enriched data
df = pd.read_csv("enriched_research_outputs_full_filtered_orcid.csv")

# FSRDC keywords (expanded list)
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

# Pre-compile regex for case-insensitive matching
FSRDC_PATTERN = re.compile('|'.join(FSRDC_KEYWORDS), flags=re.IGNORECASE)

# RDC mapping for standardization
RDC_MAPPING = {
    'michigan': 'Michigan RDC',
    'texas': 'Texas RDC',
    'boston': 'Boston RDC',
    'cornell': 'Cornell RDC',
    'berkeley': 'Berkeley RDC',
    'new york': 'NYC RDC',
    'north carolina': 'NCRDC'
}

def is_fsrdc_related(row):
    """Check if a project is FSRDC-related by scanning multiple fields."""
    # Check ProjectRDC field first
    if pd.notna(row['ProjectRDC']) and FSRDC_PATTERN.search(row['ProjectRDC']):
        return True

    # Check other fields for FSRDC keywords
    fields_to_scan = [
        'ProjectTitle',
        'OutputTitle',
        'OutputVenue',
        'OutputBiblio',
        'PI Name'
    ]

    for field in fields_to_scan:
        if pd.notna(row[field]) and FSRDC_PATTERN.search(str(row[field])):
            return True

    return False

# Apply the filter
fsrdc_df = df[df.apply(is_fsrdc_related, axis=1)].copy()

# Standardize ProjectRDC names using the mapping
for keyword, standard_name in RDC_MAPPING.items():
    fsrdc_df.loc[
        fsrdc_df['ProjectRDC'].str.contains(keyword, case=False, na=False),
        'ProjectRDC'
    ] = standard_name

# Save the filtered output
fsrdc_df.to_csv("enriched_research_outputs_fsrdc_only.csv", index=False)
print(f"Filtered data saved. Original: {len(df)} records → FSRDC-only: {len(fsrdc_df)} records")

