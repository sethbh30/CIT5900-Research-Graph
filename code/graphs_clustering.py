
#graph of entries
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the dataset (assumes 'OutputTitle', 'OutputYear', 'ProjectPI', 'Keywords' exist)
df = pd.read_csv('with_keywords.csv')
df = df[['OutputTitle', 'OutputYear', 'ProjectPI', 'Keywords']].dropna()
df = df.sample(n=150, random_state=42)

# Define a class to store entries
class Publication:
    def __init__(self, title, year, author, keywords):
        self.title = str(title)
        self.year = int(year)
        self.author = str(author)
        self.keywords = [kw.strip() for kw in str(keywords).split(',')]

    def __repr__(self):
        return f"{self.title} ({self.year})"

# Convert dataframe rows to Publication objects
publications = []
for _, row in df.iterrows():
    pub = Publication(row['OutputTitle'], row['OutputYear'], row['ProjectPI'], row['Keywords'])
    publications.append(pub)

# Build the graph
G = nx.Graph()

# Add nodes
for pub in publications:
    G.add_node(pub)

# Add edges based on shared keywords
for i, pub1 in enumerate(publications):
    for pub2 in publications[i+1:]:
        if any(k in pub2.keywords for k in pub1.keywords):
            G.add_edge(pub1, pub2, color='purple')

# Draw the graph
edge_colors = nx.get_edge_attributes(G, 'color').values()
labels = {pub: f"{pub.title[:20]}...\n{pub.year}" for pub in G.nodes}
pos = nx.spring_layout(G, k=1.5)

plt.figure(figsize=(16, 12))
nx.draw(G, pos, with_labels=True, labels=labels,
        node_size=50, font_size=8, edge_color=edge_colors)
plt.title("Graph of Publications Connected by Shared Keywords")
plt.show()

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the dataset (assumes 'OutputTitle', 'OutputYear', 'ProjectPI', 'Keywords' exist)
df = pd.read_csv('with_keywords.csv')
df = df[['OutputTitle', 'OutputYear', 'ProjectPI', 'Keywords']].dropna()

# Define a class to store entries
class Publication:
    def __init__(self, title, year, author, keywords):
        self.title = str(title)
        self.year = int(year)
        self.author = str(author)
        self.keywords = [kw.strip() for kw in str(keywords).split(',')]

    def __repr__(self):
        return f"{self.title} ({self.year})"

# Convert dataframe rows to Publication objects
publications = []
for _, row in df.iterrows():
    pub = Publication(row['OutputTitle'], row['OutputYear'], row['ProjectPI'], row['Keywords'])
    publications.append(pub)

# Build the graph
G = nx.Graph()

# Add nodes
for pub in publications:
    G.add_node(pub)

# Add edges based on shared keywords
for i, pub1 in enumerate(publications):
    for pub2 in publications[i+1:]:
        if any(k in pub2.keywords for k in pub1.keywords):
            G.add_edge(pub1, pub2, color='purple')

# Draw the graph
edge_colors = nx.get_edge_attributes(G, 'color').values()
labels = {pub: f"{pub.title[:20]}...\n{pub.year}" for pub in G.nodes}
pos = nx.spring_layout(G, k=1.5)

plt.figure(figsize=(16, 12))
nx.draw(G, pos, with_labels=True, labels=labels,
        node_size=50, font_size=8, edge_color=edge_colors)
plt.title("Graph of Publications Connected by Shared Keywords")
plt.show()

degrees = [G.degree[n] for n in G.nodes]
print(f"Average connections per publication: {sum(degrees)/len(degrees):.2f}")
print(f"Max connections: {max(degrees)}")


#clustering
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('merged_projects_with_enrichment.csv')

# Ensure columns are numeric and drop invalid rows
cols = ['ProjectStartYear', 'ProjectEndYear', 'OutputYear']
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=cols)

# Compute metrics
df['ProjectDuration'] = df['ProjectEndYear'] - df['ProjectStartYear']
df['TimeToPublishFromStart'] = df['OutputYear'] - df['ProjectStartYear']

# Filter out invalid rows
df = df[(df['ProjectDuration'] >= 0) & (df['TimeToPublishFromStart'] >= 0)].copy()

# Select features for clustering
X = df[['ProjectDuration', 'TimeToPublishFromStart']]
X_scaled = StandardScaler().fit_transform(X)

# Fit KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
df['LifecycleCluster'] = kmeans.fit_predict(X_scaled)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(df['ProjectDuration'], df['TimeToPublishFromStart'], c=df['LifecycleCluster'], cmap='tab10')
plt.xlabel('Project Duration (years)')
plt.ylabel('Time to Publish (years from start)')
plt.title('Project Lifecycle Clusters (Start to Publish)')
plt.grid(True)
plt.show()