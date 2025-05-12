
import plotly.graph_objects as go
import json

def generate_plots(data):
    """Create Plotly JSON for dashboard"""
    # Plot 1: Projects per Output Year
    fig1 = go.Figure(
        data=[go.Bar(
            x=list(data['year_counts'].keys()),
            y=list(data['year_counts'].values()),
            marker_color='#1f77b4'
        )]
    )
    fig1.update_layout(
        title='Number of Projects per Output Year',
        xaxis_title='Year',
        yaxis_title='Count'
    )
    
    # Plot 2: Active Projects Over Time
    fig2 = go.Figure(
        data=[go.Scatter(
            x=list(data['active_projects'].keys()),
            y=list(data['active_projects'].values()),
            mode='lines+markers',
            line=dict(color='#ff7f0e', width=3)
        )]
    )
    fig2.update_layout(
        title='Active Projects per Year (1990-Present)',
        xaxis_title='Year',
        yaxis_title='Number of Active Projects'
    )
    
    return {
        "plot1": fig1.to_json(),
        "plot2": fig2.to_json()
    }