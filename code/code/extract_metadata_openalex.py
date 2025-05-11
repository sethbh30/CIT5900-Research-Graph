import pandas as pd
import requests
import time
import os

# Load the CSV file
df = pd.read_csv("combined_complete_records.csv")

# Extract unique DOIs
doi_unique = df['doi'].dropna().unique()

# File to save results
output_file = "all_doi_openalex_lookup.csv"

# Track progress
buffer = []
save_every = 500  # Save after this many entries
existing_dois = set()

# Check if file exists and load existing entries (resume support)
if os.path.exists(output_file):
    existing_df = pd.read_csv(output_file)
    existing_dois = set(existing_df['doi'].str.strip().str.lower())
    print(f"Resuming... {len(existing_dois)} DOIs already processed.")
else:
    with open(output_file, "w") as f:
        f.write("doi,openalex_id,status\n")  # Write header

# Loop through all DOIs
for i, doi in enumerate(doi_unique, start=1):
    doi = doi.strip().lower()
    if doi in existing_dois:
        continue  # Skip already processed

    doi_encoded = requests.utils.quote(doi, safe='')
    url = f"https://api.openalex.org/works/https://doi.org/{doi_encoded}"

    try:
        response = requests.get(url)
        time.sleep(1)  # Respect rate limit

        if response.status_code == 200:
            data = response.json()
            openalex_id = data.get('id', '')
            status = "Found"
        elif response.status_code == 404:
            openalex_id = ""
            status = "Not Found"
        else:
            openalex_id = ""
            status = f"Error {response.status_code}"

    except Exception as e:
        openalex_id = ""
        status = f"Exception: {str(e)}"

    buffer.append({'doi': doi, 'openalex_id': openalex_id, 'status': status})

    # Save every 500 entries
    if len(buffer) >= save_every:
        pd.DataFrame(buffer).to_csv(output_file, mode='a', header=False, index=False)
        print(f"✅ Saved {len(buffer)} new records (up to DOI {i})")
        buffer = []  # Clear buffer

    if i % 500 == 0:
      print(f"Processed {i} DOIs...")

# Final save
if buffer:
    pd.DataFrame(buffer).to_csv(output_file, mode='a', header=False, index=False)
    print(f"✅ Final save: {len(buffer)} remaining records")

# Print summary
total_processed = len(existing_dois) + len(doi_unique) - len(existing_dois)
print(f"\n🚀 Finished processing {total_processed} DOIs.")


import pandas as pd
import requests
import time
import re
import os

# Load matched DOIs and OpenAlex IDs
df_dois = pd.read_csv("all_doi_openalex_lookup.csv")
df_dois = df_dois[df_dois['status'] == 'Found'].reset_index(drop=True)

# Safe getter for nested dictionary/list fields
def safe_get(obj, keys, default=None):
    for key in keys:
        if isinstance(obj, dict) and key in obj:
            obj = obj[key]
        elif isinstance(obj, list) and isinstance(key, int) and len(obj) > key:
            obj = obj[key]
        else:
            return default
    return obj

# Map OpenAlex types to desired abbreviations
output_type_map = {
    "book-chapter": "BC", "blog": "BG", "dissertation": "DI", "dataset": "DS",
    "journal-article": "JA", "mimeo": "MI", "mastersthesis": "MT",
    "report": "RE", "software": "SW", "technical-report": "TN", "working-paper": "WP"
}

# FSRDC-related keywords
fsrdc_keywords = [
    "Census Bureau", "FSRDC", "Federal Statistical Research Data Center",
    "restricted microdata", "IRS", "BEA", "confidentiality review",
    "Michigan RDC", "Texas RDC", "Boston RDC",
    "Annual Survey of Manufactures", "Census of Construction Industries",
    "Census of Finance, Insurance, and Real Estate"
]

# Initialize
enriched_results = []
fsrdc_count = 0
batch_size = 500
output_file = "fsrdc_papers_openalex.csv"
progress_interval = 500  # Print progress every 500 records

# Remove old file if exists
if os.path.exists(output_file):
    os.remove(output_file)

for i, row in df_dois.iterrows():
    # Progress tracking
    if i % progress_interval == 0:
        print(f"ℹ️ Processed {i} records so far...")

    doi = row['doi']
    openalex_id = row['openalex_id']

    if openalex_id.startswith("https://openalex.org/"):
        work_id = openalex_id.split("/")[-1]
        api_url = f"https://api.openalex.org/works/{work_id}"
    else:
        continue  # Skip invalid IDs

    try:
        response = requests.get(api_url)
        time.sleep(1)
        data = response.json()

        if not data or not isinstance(data, dict):
            continue

        # Extract metadata
        title = data.get("title", "")

        # Skip if not FSRDC-related
        if not any(keyword.lower() in title.lower() for keyword in fsrdc_keywords):
            continue

        publication_year = data.get("publication_year", "")
        publication_date = data.get("publication_date", "")
        openalex_type = data.get("type", "").lower()
        work_type = output_type_map.get(openalex_type, openalex_type.upper())

        venue_name = safe_get(data, ["host_venue", "display_name"], "")
        if not venue_name:
            venue_name = safe_get(data, ["primary_location", "source", "display_name"], "Unknown")

        biblio = data.get("biblio", {})
        volume = biblio.get("volume", "")
        issue = biblio.get("issue", "")
        pages = f"{biblio.get('first_page', '')}-{biblio.get('last_page', '')}".strip("-")

        # Detect RDC from institutions
        project_rdc = "Unknown"
        authorships = data.get("authorships", [])
        for auth in authorships:
            institutions = auth.get("institutions", [])
            for inst in institutions:
                name = inst.get("display_name", "")
                if name and (any(keyword.lower() in name.lower() for keyword in fsrdc_keywords) or
                            re.search(r'\bRDC\b', name, re.IGNORECASE)):
                    project_rdc = name
                    break
            if project_rdc != "Unknown":
                break

        project_pi = safe_get(data, ["authorships", 0, "author", "display_name"], "Unknown")

        # Extract project title from FSRDC-related title
        match = re.search(r'(FSRDC:?\s*)?([\w\s-]+?(?=using|based|analysis|study|research|$))',
                         title, re.IGNORECASE)
        project_title = match.group(2).strip() if match else title[:100]


        # Build record
        record = {
            "ProjID": i + 1,
            "ProjectStatus": "Completed",
            "ProjectTitle": project_title,
            "ProjectRDC": project_rdc,
            "ProjectYearStarted": publication_year,
            "ProjectYearEnded": publication_year,
            "ProjectPI": project_pi,
            "OutputTitle": title,
            "OutputBiblio": f"{venue_name}, Vol {volume}, Iss {issue}, pp. {pages}",
            "OutputType": work_type,
            "OutputStatus": "PB",
            "OutputVenue": venue_name,
            "OutputYear": publication_year,
            "OutputMonth": publication_date[5:7] if publication_date else "",
            "OutputVolume": volume,
            "OutputNumber": issue,
            "OutputPages": pages
        }

        enriched_results.append(record)
        fsrdc_count += 1

        # Batch saving
        if (i + 1) % batch_size == 0 or (i + 1) == len(df_dois):
            pd.DataFrame(enriched_results).to_csv(
                output_file,
                mode='a',
                header=not os.path.exists(output_file),
                index=False
            )
            print(f"💾 Saved batch of {len(enriched_results)} FSRDC papers (Total found: {fsrdc_count})")
            enriched_results = []

    except Exception as e:
        print(f"⚠️ Error processing {doi}: {str(e)}")
        continue

print(f"\n✅ Completed. Found and saved {fsrdc_count} FSRDC-related papers to {output_file}")



# Load matched DOIs and OpenAlex IDs
df_dois = pd.read_csv("all_doi_openalex_lookup.csv")
df_dois = df_dois[df_dois['status'] == 'Found'].reset_index(drop=True)

# Configuration - MODIFY THESE VALUES
start_from_record = 13000  # Start from where you left off
output_file = "fsrdc_papers_openalex.csv"
existing_fsrdc_count = 628  # Current count from your output

# Check if output file exists to determine if we need headers
file_exists = os.path.exists(output_file)

# [Keep all your existing functions and constants...]

# Initialize with existing count
fsrdc_count = existing_fsrdc_count
enriched_results = []
batch_size = 500
progress_interval = 500

# Process only remaining records
for i in range(start_from_record, len(df_dois)):
    row = df_dois.iloc[i]

    # Progress tracking
    if i % progress_interval == 0:
        print(f"ℹ️ Processed {i} records so far...")

    doi = row['doi']
    openalex_id = row['openalex_id']

    if openalex_id.startswith("https://openalex.org/"):
        work_id = openalex_id.split("/")[-1]
        api_url = f"https://api.openalex.org/works/{work_id}"
    else:
        continue

    try:
        response = requests.get(api_url)
        time.sleep(1)
        data = response.json()

        if not data or not isinstance(data, dict):
            continue

        # [Keep all your existing processing logic...]

        enriched_results.append(record)
        fsrdc_count += 1

        # Batch saving
        if (i + 1) % batch_size == 0 or (i + 1) == len(df_dois):
            pd.DataFrame(enriched_results).to_csv(
                output_file,
                mode='a',
                header=not file_exists,
                index=False
            )
            print(f"💾 Saved batch of {len(enriched_results)} FSRDC papers (Total found: {fsrdc_count})")
            enriched_results = []
            file_exists = True  # After first write, headers exist

    except Exception as e:
        print(f"⚠️ Error processing {doi}: {str(e)}")
        continue

print(f"\n✅ Completed. Found and saved {fsrdc_count} FSRDC-related papers to {output_file}")