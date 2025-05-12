fetch('data/publications.json')
    .then(response => response.json())
    .then(data => {
        // Render Plot 1
        Plotly.newPlot(
            'projects-per-year',
            JSON.parse(data.plots.plot1).data,
            JSON.parse(data.plots.plot1).layout
        );
        
        // Render Plot 2
        Plotly.newPlot(
            'active-projects-trend',
            JSON.parse(data.plots.plot2).data,
            JSON.parse(data.plots.plot2).layout
        );
        
        // Make plots responsive
        window.addEventListener('resize', function() {
            Plotly.Plots.resize('projects-per-year');
            Plotly.Plots.resize('active-projects-trend');
        });
    });