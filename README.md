# FSRDC Research Output Inventory and Analysis

## Project Overview
This repository contains our (Group 5's) work for CIT 5900-002's three-part project on Federal Statistical Research Data Centers (FSRDC) research output analysis. The project progressed from basic data modeling to a comprehensive research output analysis platform with visualization dashboard.

## Project Journey

### Project 1: Data Modeling and Basic Graph Analysis
- Developed foundational data structures to represent FSRDC research outputs
- Implemented graph data structures where nodes represented research outputs
- Created connections between outputs based on shared attributes (agencies, keywords, etc.)
- Implemented BFS/DFS algorithms for graph traversal and search functionality
- Focus was on Python programming fundamentals: file I/O, OOP, data structures, and graph algorithms

### Project 2: Web Scraping, API Integration, and Advanced Analysis
- Expanded the dataset through web scraping of public FSRDC project information
- Integrated with research APIs (OpenAlex, NIH, BASE, CORE) to gather metadata
- Applied entity resolution techniques to identify and remove duplicates
- Implemented data validation to ensure outputs met FSRDC criteria
- Enhanced graph analysis using NetworkX for centrality measures and clustering
- Applied data analysis techniques using Pandas and visualization libraries

### Project 3: Comprehensive Analysis and Web Dashboard
- Consolidated and deduplicated data from multiple group sources
- Performed extensive data enrichment to standardize records according to required schema:
  - Project metadata (ID, Status, Title, RDC, Year Started/Ended, PI)
  - Output details (Title, Bibliography, Type, Status, Venue, Publication metadata)
- Conducted exploratory data analysis to identify:
  - Top 10 performing RDCs by research output volume
  - Publication trends over time
  - Most prolific FSRDC authors
  - Citation impact analysis
- Applied advanced data science techniques:
  - Regression/classification models
  - Principal Component Analysis (PCA)
  - Clustering algorithms
  - Text processing methods
- Created this GitHub Pages site to visualize findings and provide an interactive dashboard

## Project Structure

```bash
CIT5900-Publications-Pipeline/
├── code/                      # Python backend
│   ├── analysis/
│   │   ├── data_statistics.py
│   │   └── graphs_clustering.py
│   ├── metadata_extraction/   # API scrapers (nih.py, openalex.py)
│   │   ├── extract_metadata_core.py
│   │   ├── extract_metadata_nih.py
│   │   ├── extract_metadata_openalex.py
│   │   └── extract_metadata_orcid.py
│   ├── processing/
│   │   └── deduplication_data_enrichment.py
│   ├── visualization/
│   │   └── visualization.py
│   └── tests/
│       └── test_extract_metadata.py
├── docs/                      # GitHub Pages dashboard (HTML/JS/CSS)
│   ├── data/                  # Final Output dataset
        ├──ResearchOutputs_Group5.csv
│   ├── images/
│   │   ├── plot1.png
│   │   └── plot2.png
│   ├── config.yml
│   ├── index.html
│   ├── dashboard.js
│   └── style.css
├── main.py                    # Master script to run the project
├── requirements.txt
└── README.md
```

## Technologies Used
- **Python**: Core programming language
- **Pandas/NumPy**: Data processing and numerical analysis
- **NetworkX**: Graph construction and analysis
- **BeautifulSoup/Requests**: Web scraping components
- **Scikit-learn**: Machine learning implementations
- **Matplotlib/Plotly**: Data visualization
- **GitHub Pages**: Web dashboard hosting

## Repository Structure
- `main.py`: Entry point for running analysis code
- `data/`: Contains processed datasets
- `code/`: Source code for data processing and analysis
- `docs/`: Documentation and GitHub Pages files
- `tests/`: Unit tests for code validation

## Group Members
- Aditi Madathingal
- Nikola Datkova
- Bhavika Seth

## How to Use
1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Run the main analysis: `python main.py`
4. If you want to run individual components:
`python src/metadata_extraction/nih.py`      
`python src/processing/deduplication.py`    
`python src/visualization.py`
5. View the dashboard at (https://github.com/sethbh30/FSRDC-Publications-Pipeline)

## Contributing
Pull requests welcome! For major changes, open an issue first.