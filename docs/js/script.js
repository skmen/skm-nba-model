$(document).ready(function() {
    // Initialize DataTable with empty data first
    var table = $('#predictionsTable').DataTable({
        scrollX: true, // Enable horizontal scrolling for many columns
        pageLength: 25,
        columns: [
            { data: 'PLAYER_NAME' },
            { data: 'TEAM_NAME' },
            { 
                data: 'IS_HOME',
                render: function(data, type, row) {
                    // Render TRUE as "Home" and FALSE as "Away"
                    return (data === 'True' || data === true) ? '🏠 Home' : '✈️ Away';
                }
            },
            { data: 'MINUTES' },      // New Column
            { data: 'USAGE_RATE' },   // New Column
            { data: 'PACE' },         // New Column
            { data: 'OPP_DvP' },      // New Column
            { data: 'PTS' },
            { data: 'REB' },
            { data: 'AST' },
            { data: 'STL' },
            { data: 'BLK' },
            { data: 'PRA' }           // New Column
        ],
        order: [[7, 'desc']] // Default sort by PTS (Index 7) descending
    });

    // Listen for file upload
    $('#csvFileInput').change(function(e) {
        var file = e.target.files[0];
        if (!file) return;

        Papa.parse(file, {
            header: true,
            dynamicTyping: true, // Automatically convert numbers
            skipEmptyLines: true,
            complete: function(results) {
                // Clear existing data
                table.clear();

                // Check if parsing worked
                if (results.data.length === 0) {
                    console.error("CSV is empty or could not be parsed.");
                    return;
                }

                // Add new data
                table.rows.add(results.data).draw();
            },
            error: function(error) {
                console.error("Error parsing CSV:", error);
                alert("Failed to load CSV file.");
            }
        });
    });
});