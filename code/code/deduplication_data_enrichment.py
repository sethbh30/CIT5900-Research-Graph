import pandas as pd

# Define column priorities
column_mapping = {
    'author': ['author', 'pi', 'researchers', 'project_pi', 'projectpi'],
    'year': ['year', 'projectendyear'],
    'title': ['title', 'projecttitle'],
    'doi': ['doi']
}

dataframes = []

for i in range(1, 9):
    file_name = f"group{i}.csv"
    try:
        df = pd.read_csv(file_name)
        lower_columns = {col.lower(): col for col in df.columns}
        new_df = pd.DataFrame()

        for target_col, source_options in column_mapping.items():
            collected = []

            for source_col in source_options:
                lower_col = source_col.lower()
                if lower_col in lower_columns:
                    actual_col = lower_columns[lower_col]
                    # Extract and clean values
                    cleaned = (
                        df[actual_col]
                        .astype(str)
                        .str.split(r'[;,]')
                        .apply(lambda parts: [part.strip() for part in parts])
                    )
                    collected.append(cleaned)

            if collected:
                # Combine all matched columns row-wise and take the first non-null value per row
                new_df[target_col] = pd.DataFrame(collected).T.apply(
                    lambda row: next((item for sublist in row if isinstance(sublist, list) for item in sublist if item and item.lower() != 'nan'), pd.NA),
                    axis=1
                )
            else:
                new_df[target_col] = pd.NA

        # Filter out rows where any of the required columns are empty/NA
        complete_records = new_df.dropna(subset=column_mapping.keys())

        if not complete_records.empty:
            dataframes.append(complete_records)

    except FileNotFoundError:
        print(f"Warning: {file_name} not found. Skipping.")

# Combine and save
if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_csv("combined_complete_records.csv", index=False)
    print(f"Saved {len(combined_df)} complete records.")
else:
    print("No complete records found.")