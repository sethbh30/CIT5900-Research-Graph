#NIH PMC
import pandas as pd
import requests
import time
import os
from xml.etree import ElementTree

# NIH E-utilities API endpoint
ncbi_esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

# Output file
output_file = "doi_pubmed_lookup.csv"
save_every = 500

# Load full list
df = pd.read_csv("combined_complete_records.csv")
doi_list = df['doi'].dropna().unique()

# Load already processed DOIs (if resuming)
existing_dois = set()
if os.path.exists(output_file):
    existing_df = pd.read_csv(output_file)
    existing_dois = set(existing_df['doi'].str.strip().str.lower())
    print(f"🔁 Resuming from previous run: {len(existing_dois)} DOIs already processed.")
else:
    with open(output_file, "w") as f:
        f.write("doi,status,pmid\n")  # Write header

# Prepare buffer for incremental saving
buffer = []

# Start processing
for i, doi in enumerate(doi_list, start=1):
    doi = doi.strip().lower()
    if doi in existing_dois:
        continue  # Skip already processed

    params = {
        "db": "pubmed",
        "term": doi,
        "retmode": "xml"
    }

    try:
        response = requests.get(ncbi_esearch_url, params=params)
        time.sleep(0.4)

        if response.status_code == 200:
            root = ElementTree.fromstring(response.content)
            id_list = root.find("IdList")
            pmids = id_list.findall("Id") if id_list is not None else []

            if len(pmids) > 0:
                buffer.append({"doi": doi, "status": "Found", "pmid": pmids[0].text})
            else:
                buffer.append({"doi": doi, "status": "Not Found", "pmid": None})
        else:
            buffer.append({"doi": doi, "status": f"Error {response.status_code}", "pmid": None})

    except Exception as e:
        buffer.append({"doi": doi, "status": "Exception", "pmid": None})

    # Save every 500 entries
    if len(buffer) >= save_every:
        pd.DataFrame(buffer).to_csv(output_file, mode='a', header=False, index=False)
        print(f"💾 Saved {len(buffer)} records at DOI {i}")
        buffer = []

    if i % 500 == 0:
      print(f"Processed {i} DOIs...")


# Final save
if buffer:
    pd.DataFrame(buffer).to_csv(output_file, mode='a', header=False, index=False)
    print(f"✅ Final save of {len(buffer)} remaining records.")

# Summary
df_final = pd.read_csv(output_file)
total = len(df_final)
matched = len(df_final[df_final["status"] == "Found"])
print(f"\n🎯 PubMed search complete.")
print(f"   🔍 Total DOIs checked: {total}")
print(f"   ✅ Total matched in PubMed: {matched}")
print(f"   ❌ Not found: {total - matched}")


#extraction of pmcid
import pandas as pd
import requests
import time
from xml.etree import ElementTree
from tqdm import tqdm  # for progress bars

# Configure
API_DELAY = 0.34  # NIH rate limit (3 requests/second)
PMC_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
OUTPUT_FILE = "pmc_availability_results_with_doi.csv"

# Load your data
df = pd.read_csv("combined_complete_records.csv")
print(f"Loaded {len(df)} records from your file")

def clean_author(author_str):
    """Extract first author's last name from author string"""
    if pd.isna(author_str):
        return ""
    first_author = author_str.split(";")[0].split(",")[0].strip()
    return first_author.split()[0] if first_author else ""

def check_by_doi(doi):
    """Check PMC availability by DOI (most reliable method)"""
    if pd.isna(doi) or not str(doi).strip():
        return False, None

    clean_doi = str(doi).strip()
    params = {
        "db": "pmc",
        "term": f'{clean_doi}[DOI]',
        "retmode": "xml",
        "retmax": 1
    }

    try:
        response = requests.get(PMC_SEARCH_URL, params=params)
        time.sleep(API_DELAY)

        if response.status_code == 200:
            root = ElementTree.fromstring(response.content)
            count = int(root.findtext(".//Count") or "0")
            if count > 0:
                pmcid = root.findtext(".//IdList/Id")
                return True, pmcid
    except Exception as e:
        print(f"Error checking DOI {clean_doi[:30]}...: {str(e)}")

    return False, None

def check_by_title_author(title, author):
    """Fallback method: Check by title and author"""
    query = f'("{title}"[Title]) AND {author}[Author]'

    params = {
        "db": "pmc",
        "term": query,
        "retmode": "xml",
        "retmax": 1
    }

    try:
        response = requests.get(PMC_SEARCH_URL, params=params)
        time.sleep(API_DELAY)

        if response.status_code == 200:
            root = ElementTree.fromstring(response.content)
            count = int(root.findtext(".//Count") or "0")
            if count > 0:
                pmcid = root.findtext(".//IdList/Id")
                return True, pmcid
    except Exception as e:
        print(f"Error checking {title[:30]}...: {str(e)}")

    return False, None

# Prepare results
results = []

# Process records with progress bar
for idx, row in tqdm(df.iterrows(), total=len(df)):
    title = str(row['title'])[:500]  # truncate very long titles
    author = clean_author(str(row.get('author', '')))
    doi = row.get('doi', '')  # Assuming your DOI column is named 'doi'

    # Initialize result dict
    result = {
        'ProjID': idx,
        'Title': title,
        'Author': author,
        'DOI': doi,
        'In_PMC': False,
        'PMCID': None,
        'Method': None,
        'Reason': None
    }

    # First try DOI lookup if available
    if doi and str(doi).strip() and str(doi).strip().lower() != 'nan':
        found, pmcid = check_by_doi(doi)
        if found:
            result.update({
                'In_PMC': True,
                'PMCID': pmcid,
                'Method': 'DOI',
                'Reason': 'Found by DOI'
            })
            results.append(result)
            continue

    # Fall back to title/author if no DOI or DOI lookup failed
    if title and author:
        found, pmcid = check_by_title_author(title, author)
        if found:
            result.update({
                'In_PMC': True,
                'PMCID': pmcid,
                'Method': 'Title/Author',
                'Reason': 'Found by title/author'
            })
        else:
            result.update({
                'Reason': 'Not found in PMC'
            })
    else:
        result.update({
            'Reason': 'Missing title or author'
        })

    results.append(result)

# Save enhanced results
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_FILE, index=False)
print(f"Saved results to {OUTPUT_FILE}")

# Print summary
print("\n=== Summary of findings ===")
print(f"Total records processed: {len(df_results)}")
print(f"Records found in PMC: {len(df_results[df_results['In_PMC']])}")
print("\nBreakdown by discovery method:")
print(df_results[df_results['In_PMC']]['Method'].value_counts())
print("\nReasons for not finding records:")
print(df_results[~df_results['In_PMC']]['Reason'].value_counts())

# Load previous results
df_availability = pd.read_csv("pmc_availability_results_with_doi.csv")
pmc_papers = df_availability[df_availability['In_PMC']].copy()
print(f"Found {len(pmc_papers)} papers in PMC to process")

#Metadata extraction
import pandas as pd
import requests
import time
import re
from xml.etree import ElementTree
from tqdm import tqdm

# Keywords for FSRDC-related filtering
fsrdc_keywords = [
    "Census Bureau", "FSRDC", "Federal Statistical Research Data Center",
    "restricted microdata", "IRS", "BEA", "confidentiality review",
    "Michigan RDC", "Texas RDC", "Boston RDC",
    "Annual Survey of Manufactures", "Census of Construction Industries",
    "Census of Finance, Insurance, and Real Estate"
]

# Output type mapping
output_type_map = {
    "book-chapter": "BC", "blog": "BG", "dissertation": "DI", "dataset": "DS",
    "journal-article": "JA", "mimeo": "MI", "mastersthesis": "MT",
    "report": "RE", "software": "SW", "technical-report": "TN", "working-paper": "WP"
}

def is_fsrdc_related(text):
    if not isinstance(text, str):
        return False
    return any(keyword.lower() in text.lower() for keyword in fsrdc_keywords)

def fetch_pmc_metadata(pmcid):
    try:
        params = {"db": "pmc", "id": pmcid, "retmode": "xml"}
        response = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", params=params)
        time.sleep(0.34)
        return response.text if response.status_code == 200 else None
    except Exception:
        return None

def parse_pmc_metadata(xml_content):
    try:
        root = ElementTree.fromstring(xml_content)
        article = root.find('.//article')
        if article is None:
            return None

        title = " ".join([t.text for t in article.findall('.//article-title') if t.text])
        abstract = " ".join([t.text for t in article.findall('.//abstract//p') if t.text])

        if not (is_fsrdc_related(title) or is_fsrdc_related(abstract)):
            return None

        first_author = "Unknown"
        institution = "Unknown"

        for contrib in article.findall('.//contrib[@contrib-type="author"]'):
            name = contrib.find('.//name')
            if name is not None:
               surname = name.findtext('surname') or ""
               given_names = name.findtext('given-names') or ""
               full_name = f"{given_names} {surname}".strip()
               if full_name:
                  first_author = full_name

            aff = contrib.find('.//aff')
            if aff is not None and aff.text:
               institution = aff.text.strip()

            break  # Only process the first author



        journal_title = article.findtext('.//journal-title-group/journal-title')
        pub_date = article.find('.//pub-date[@date-type="pub"]')
        year = pub_date.findtext('year') if pub_date is not None else ''
        month = pub_date.findtext('month') if pub_date is not None else ''
        volume = article.findtext('.//volume') or ''
        issue = article.findtext('.//issue') or ''
        fpage = article.findtext('.//fpage')
        lpage = article.findtext('.//lpage')
        pages = f"{fpage}-{lpage}" if fpage and lpage else fpage or ''

        article_type = article.get('article-type', '').lower()

        return {
            'Title': title,
            'Year': year,
            'Month': month,
            'Volume': volume,
            'Issue': issue,
            'Pages': pages,
            'ArticleType': article_type,
            'Journal': journal_title,
            'ProjectPI': first_author,
            'Institution': institution
        }

    except Exception:
        return None

def process_nih_records(pmc_ids, output_file="fsrdc_papers_pmc.csv"):
    results = []
    total_saved = 0
    save_interval = 500

    for i, pmcid in enumerate(tqdm(pmc_ids, desc="Processing PMC records"), start=1):
        xml_content = fetch_pmc_metadata(pmcid)
        if not xml_content:
            continue

        metadata = parse_pmc_metadata(xml_content)
        if metadata:
            row = metadata
            output_type = output_type_map.get(row.get('ArticleType', '').lower(), 'OT')

            journal = str(row.get('Journal', '')).strip()
            volume = str(row.get('Volume', '')).strip()
            issue = str(row.get('Issue', '')).strip()
            pages = str(row.get('Pages', '')).strip()

            if all([journal, volume, issue, pages]):
                full_citation = f"{journal}, Vol {volume}, Iss {issue}, pp. {pages}"
            else:
                full_citation = "None"

            results.append({
                'ProjID': total_saved + len(results) + 1,
                'ProjectStatus': 'Completed',
                'ProjectTitle': row.get('Title', '') if "fsrdc" in row.get('Title', '').lower() else "FSRDC Project",
                'ProjectRDC': row.get('Institution', 'Unknown'),
                'ProjectYearStarted': row.get('Year', 'Unknown'),
                'ProjectYearEnded': row.get('Year', 'Unknown'),
                'ProjectPI': row.get('ProjectPI', 'Unknown'),
                'OutputTitle': row.get('Title', 'Unknown'),
                'OutputBiblio': full_citation,
                'OutputType': output_type,
                'OutputStatus': 'PB',
                'OutputVenue': journal,
                'OutputYear': row.get('Year', 'Unknown'),
                'OutputMonth': str(row.get('Month', 'Unknown'))[:2],
                'OutputVolume': volume,
                'OutputNumber': issue,
                'OutputPages': pages
            })

        # Print processing update every 100 records
        if i % 100 == 0:
            print(f"🔄 Processed {i} PMC records...")

        # Save every 500 results to the same file
        if len(results) >= save_interval:
            df = pd.DataFrame(results)
            mode = 'a' if total_saved > 0 else 'w'
            header = total_saved == 0
            df.to_csv(output_file, mode=mode, index=False, header=header)
            print(f"💾 Saved {len(df)} more records to {output_file}")
            total_saved += len(df)
            results.clear()

    # Save remaining entries
    if results:
        df = pd.DataFrame(results)
        mode = 'a' if total_saved > 0 else 'w'
        header = total_saved == 0
        df.to_csv(output_file, mode=mode, index=False, header=header)
        print(f"💾 Saved final {len(df)} records to {output_file}")
        total_saved += len(df)

    print(f"✅ Total FSRDC-related papers found: {total_saved}")
    return total_saved


# Example run:
pmc_ids = pd.read_csv("pmc_availability_results_with_doi.csv")['PMCID'].dropna().tolist()
process_nih_records(pmc_ids)
