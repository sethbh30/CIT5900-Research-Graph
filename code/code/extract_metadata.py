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
