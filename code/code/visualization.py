import pandas as pd
import plotly.express as px

# Load your data
df = pd.read_csv("doi_rdc_filtered_only_true.csv", dtype=str)

# Convert year columns to numeric
df["ProjectYearStarted"] = pd.to_numeric(df["ProjectYearStarted"], errors="coerce")
df["ProjectYearEnded"] = pd.to_numeric(df["ProjectYearEnded"], errors="coerce")

# Drop rows with invalid years
df = df.dropna(subset=["ProjectYearStarted", "ProjectYearEnded"])

# Count active projects per year
active_years = {}

for _, row in df.iterrows():
    start = int(row["ProjectYearStarted"])
    end = int(row["ProjectYearEnded"])
    for year in range(start, end + 1):
        active_years[year] = active_years.get(year, 0) + 1

# Convert to DataFrame
year_df = pd.DataFrame(sorted(active_years.items()), columns=["Year", "ActiveProjects"])

# Filter to years from 1990 onward
year_df = year_df[year_df["Year"] >= 1990]

# Create interactive line plot
fig = px.line(year_df, x="Year", y="ActiveProjects",
              title="Active Projects per Year (1990–Present)",
              markers=True)

fig.update_layout(
    xaxis_title="Year",
    yaxis_title="Number of Active Projects",
    template="plotly_white"
)

fig.show()

#Number of projects per year
from datetime import datetime

# Get current year
current_year = datetime.now().year

# Print active project count for each year from 1990 to now
for year in range(1990, current_year + 1):
    count = active_years.get(year, 0)
    print(f"{year}: {count} active projects")

import plotly.express as px

# Prepare the same counts
year_counts = df["OutputYear"].value_counts().sort_index().reset_index()
year_counts.columns = ["OutputYear", "Count"]

# Plot
fig = px.bar(year_counts, x="OutputYear", y="Count",
             title="Number of Projects per Output Year",
             labels={"OutputYear": "Year", "Count": "Projects"})

fig.update_layout(xaxis_tickangle=-45)
fig.show()

#output year countsx
import pandas as pd

df = pd.read_csv("doi_rdc_filtered_only_true.csv", dtype=str)  # replace with actual file name
df["OutputYear"] = pd.to_numeric(df["OutputYear"], errors="coerce")
output_year_counts = df["OutputYear"].value_counts().sort_index()

print(output_year_counts)

#Top publishing PI countsx
import pandas as pd

# Load your CSV
df = pd.read_csv("doi_rdc_filtered_only_true.csv", dtype=str)  # replace with actual filename

# Count outputs per author
top_authors = df["ProjectPI"].value_counts().head(20)

print(top_authors)