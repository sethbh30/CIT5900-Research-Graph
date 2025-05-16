// Load visualization data
fetch('data/visualizations.json')
    .then(response => response.json())
    .then(data => {
        // Render Active Projects Timeline
        Plotly.newPlot(
            'active-projects-container',
            JSON.parse(data.active).data,
            JSON.parse(data.active).layout
        );
        
        // Render Publications by Year
        Plotly.newPlot(
            'publications-container',
            JSON.parse(data.publications).data,
            JSON.parse(data.publications).layout
        );
        
        // Make responsive
        window.addEventListener('resize', function() {
            Plotly.Plots.resize('active-projects-container');
            Plotly.Plots.resize('publications-container');
        });
    });