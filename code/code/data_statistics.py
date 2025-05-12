import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_pca_on_topic_trends(file_path, n_components=18):
    # Load the data
    df = pd.read_csv(file_path, dtype=str)

    # Clean and prepare
    df = df[['OutputYear', 'PredictedTopic']].dropna()
    # df = df[df['PredictedTopic'] != 'economics']
    df['OutputYear'] = pd.to_numeric(df['OutputYear'], errors='coerce')
    df = df.dropna()

    # Create year-topic count matrix
    matrix = df.pivot_table(index='OutputYear',
                            columns='PredictedTopic',
                            aggfunc=len,
                            fill_value=0)
    print(matrix)

    # Standardize the data
    scaler = StandardScaler()
    matrix_std = scaler.fit_transform(matrix)

    # PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(matrix_std)

    eigenvals = pca.explained_variance_ratio_
    print("Eigenvalues:", eigenvals)
    print(sum(eigenvals))

    # Wrap result in DataFrame
    pca_df = pd.DataFrame(components,
                          index=matrix.index,
                          columns=[f'PC{i+1}' for i in range(n_components)])

    return pca_df, pca


# Assuming your DataFrame is already loaded as df
df = pd.read_csv('merged_projects_with_enrichment.csv', dtype=str)
topic_counts = df["PredictedTopic"].value_counts()

# Print the counts
print(topic_counts.head(10))

import pandas as pd
import numpy as np

# Load the original file with topic and year data
df = pd.read_csv("merged_projects_with_enrichment.csv")

# Display column names to ensure we're using the correct ones
print("Available columns:")
print(df.columns.tolist())

# Try different possible column names for year and topic
year_columns = ['OutputYear', 'Year', 'output_year', 'year', 'YEAR', 'Publication_Year']
topic_columns = ['PredictedTopic', 'Topic', 'predicted_topic', 'topic', 'TOPIC', 'Topic_ID']

# Find the actual year column and topic column
year_col = None
for col in year_columns:
    if col in df.columns:
        year_col = col
        break

topic_col = None
for col in topic_columns:
    if col in df.columns:
        topic_col = col
        break

if not year_col:
    print("ERROR: Couldn't find a year column. Please specify the correct column name.")
    # List the columns that look like they might contain years
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    print("Potential year columns (numeric):", numeric_cols.tolist())
    # Try to identify a year column by checking values
    for col in numeric_cols:
        if df[col].min() >= 1900 and df[col].max() <= 2030:
            print(f"Possible year column: {col} (range: {df[col].min()} to {df[col].max()})")
    year_col = input("Enter the year column name: ")

if not topic_col:
    print("ERROR: Couldn't find a topic column. Please specify the correct column name.")
    # List string columns that might be topic columns
    object_cols = df.select_dtypes(include=['object']).columns
    print("Potential topic columns (string):", object_cols.tolist())
    topic_col = input("Enter the topic column name: ")

print(f"Using '{year_col}' as year column and '{topic_col}' as topic column")

# Filter out rows with missing or invalid values
df_filtered = df.copy()

# Filter for numeric years and convert to integers
df_filtered = df_filtered[pd.to_numeric(df_filtered[year_col], errors='coerce').notna()]
df_filtered[year_col] = df_filtered[year_col].astype(int)

# Filter out missing topics
df_filtered = df_filtered[df_filtered[topic_col].notna()]

# Filter for years >= 1990
df_filtered = df_filtered[df_filtered[year_col] >= 1990]

print(f"After filtering: {len(df_filtered)} rows")

# Check if we have data to work with
if len(df_filtered) == 0:
    print("ERROR: No data left after filtering. Check your column names and data.")
else:
    # Create the pivot table
    try:
        topic_year_matrix = df_filtered.pivot_table(
            index=topic_col,
            columns=year_col,
            aggfunc='size',
            fill_value=0
        )

        # Display matrix information
        print(f"\nMatrix shape: {topic_year_matrix.shape}")
        print(f"Year range: {topic_year_matrix.columns.min()} to {topic_year_matrix.columns.max()}")
        print(f"Number of topics: {len(topic_year_matrix.index)}")

        # Display at least a sample of the matrix
        print("\nSample of Topic-Year Matrix (first 10 rows, all columns):")
        print(topic_year_matrix.head(10))

        # For large matrices, also show a more compact view
        if topic_year_matrix.shape[0] > 20 or topic_year_matrix.shape[1] > 10:
            print("\nCompact view - Top 10 topics by frequency:")
            topic_sums = topic_year_matrix.sum(axis=1).sort_values(ascending=False)
            top_topics = topic_sums.head(10).index
            recent_years = sorted(topic_year_matrix.columns)[-5:]
            print(topic_year_matrix.loc[top_topics, recent_years])

        # Show row and column totals
        print("\nTotal by topic (top 10):")
        print(topic_year_matrix.sum(axis=1).sort_values(ascending=False).head(10))

        print("\nTotal by year:")
        print(topic_year_matrix.sum(axis=0))

        # Save to CSV for easy viewing
        matrix_filename = 'topic_year_matrix.csv'
        topic_year_matrix.to_csv(matrix_filename)
        print(f"\nFull matrix saved to '{matrix_filename}'")

    except Exception as e:
        print(f"Error creating matrix: {str(e)}")
        print("\nTrying to troubleshoot...")

        # Show a summary of the filtered data to help debug
        print(f"\nSummary of filtered data:")
        print(f"Year column ({year_col}) values: {df_filtered[year_col].value_counts().head()}")
        print(f"Topic column ({topic_col}) value counts: {df_filtered[topic_col].value_counts().head()}")

        # Try with a simpler approach
        print("\nAttempting a simpler crosstab approach:")
        try:
            simple_matrix = pd.crosstab(
                index=df_filtered[topic_col],
                columns=df_filtered[year_col],
                dropna=False
            )
            print(simple_matrix.head(10))
        except Exception as e2:
            print(f"Error with crosstab approach: {str(e2)}")

pca_result, pca_model = run_pca_on_topic_trends("merged_projects_with_enrichment.csv")
print(pca_result.head(40))

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def create_interactive_pca_plot(pca_df):
    # Reset index to make year a column
    plot_df = pca_df.reset_index()

    # Create interactive plot with Plotly
    fig = px.scatter(
        plot_df,
        x='PC1',
        y='PC2',
        color='OutputYear',
        color_continuous_scale='Viridis',
        hover_name='OutputYear',
        title='Interactive PCA of Topic Trends (1949-2025)',
        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
        size_max=15,
        size=[10] * len(plot_df)  # Uniform size
    )

    # Add trajectory lines
    sorted_df = plot_df.sort_values('OutputYear')
    fig.add_trace(
        go.Scatter(
            x=sorted_df['PC1'],
            y=sorted_df['PC2'],
            mode='lines',
            line=dict(color='rgba(0,0,0,0.2)', width=1),
            showlegend=False
        )
    )

    # Add year labels
    fig.add_trace(
        go.Scatter(
            x=plot_df['PC1'],
            y=plot_df['PC2'],
            mode='text',
            text=plot_df['OutputYear'].astype(int).astype(str),
            textposition="middle center",
            textfont=dict(color='black', size=10),
            showlegend=False
        )
    )

    # Customize layout
    fig.update_layout(
        width=900,
        height=700,
        template='plotly_white',
        xaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='gray'),
        yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='gray')
    )

    return fig

# Create and display the interactive plot
interactive_plot = create_interactive_pca_plot(pca_result)
interactive_plot.show()