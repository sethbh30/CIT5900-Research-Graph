import plotly.express as px
import json
from pathlib import Path

def generate_interactive_plots():
    # Active Projects Timeline (1990-2025)
    active_data = {
        "Year": list(range(1990, 2026)),
        "Count": [
            0, 0, 0, 1, 2, 4, 5, 6, 5, 4, 4, 4, 7, 6, 9, 9, 12, 19, 19, 24,
            21, 18, 25, 28, 33, 46, 54, 57, 60, 48, 38, 30, 32, 40, 40, 35
        ]
    }

    # Projects per Output Year
    output_data = {
        "Year": [1949, 1964] + list(range(1993, 2026)),
        "Count": [1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 2, 6, 3, 6, 5, 4,
                 7, 11, 5, 16, 20, 16, 13, 8, 7, 6, 15, 8, 16, 4]
    }

    # Create plots
    fig_active = px.line(
        active_data,
        x="Year",
        y="Count",
        title="Active Projects per Year (1990-2025)",
        markers=True
    ).update_layout(
        yaxis_title="Active Projects",
        hovermode="x unified"
    )

    fig_output = px.bar(
        output_data,
        x="Year",
        y="Count",
        title="Projects Published per Year",
        text="Count"
    ).update_layout(
        xaxis_tickangle=45,
        yaxis_title="Publications"
    )

    # Save to JSON
    output = {
        "plots": {
            "active_timeline": fig_active.to_json(),
            "publications_by_year": fig_output.to_json()
        },
        "stats": {
            "peak_active": {"year": 2018, "value": 60},
            "peak_publications": {"year": 2016, "value": 20},
            "current_active": 35
        }
    }

    Path("docs/data").mkdir(exist_ok=True)
    with open("docs/data/publications.json", "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    generate_interactive_plots()