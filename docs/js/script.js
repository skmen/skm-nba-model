$(document).ready(function() {

    // =========================================================
    // 1. GAME PREDICTIONS JSON LOGIC (Scoreboard)
    // =========================================================
    // We put this first so it works even if the table fails
    $('#jsonFileInput').change(function(e) {
        var file = e.target.files[0];
        if (!file) return;

        var reader = new FileReader();
        reader.onload = function(event) {
            try {
                var games = JSON.parse(event.target.result);
                renderGameCards(games);
            } catch (err) {
                console.error("Error parsing JSON:", err);
                alert("Invalid JSON file");
            }
        };
        reader.readAsText(file);
    });

    function renderGameCards(games) {
        var container = $('#gamePredictionsContainer');
        container.empty();

        games.forEach(function(game) {
            var awayClass = game.winner === game.away_team ? 'winner' : '';
            var homeClass = game.winner === game.home_team ? 'winner' : '';

            var cardHtml = `
                <div class="score-card">
                    <h3>${game.away_team} @ ${game.home_team}</h3>
                    <div class="score-row ${awayClass}">
                        <span>${game.away_team}</span>
                        <span>${game.away_score}</span>
                    </div>
                    <div class="score-row ${homeClass}">
                        <span>${game.home_team}</span>
                        <span>${game.home_score}</span>
                    </div>
                    <div class="meta-info">
                        Spread: ${game.winner} -${game.spread} <br>
                        Total: ${game.total} | Pace: ${game.pace}
                    </div>
                </div>
            `;
            container.append(cardHtml);
        });
    }

    // =========================================================
    // 2. PAGE 1: PREDICTIONS VIEWER (index.html)
    // =========================================================
    if ($('#predictionsTable').length) {
        var viewerTable = $('#predictionsTable').DataTable({
            scrollX: true,
            pageLength: 25,
            columns: [
                // Make sure these match your CSV headers EXACTLY
                { data: 'PLAYER_NAME' },
                { data: 'TEAM_NAME' },
                { 
                    data: 'IS_HOME',
                    defaultContent: "", // Prevents crash if empty
                    render: function(data) {
                        return (data === 'True' || data === true) ? '🏠 Home' : '✈️ Away';
                    }
                },
                { data: 'MINUTES', defaultContent: 0 },
                { data: 'USAGE_RATE', defaultContent: 0 },
                { data: 'PACE', defaultContent: 0 },
                { data: 'OPP_DvP', defaultContent: 1.0 },
                { data: 'PTS' },
                { data: 'REB' },
                { data: 'AST' },
                { data: 'STL' },
                { data: 'BLK' },
                { data: 'PRA' }
            ],
            // Removed PREDICTION_TIME from columns list above
            order: [[7, 'desc']] 
        });

        // Specific upload handler for Index page
        $('#csvFileInput').change(function(e) {
            handleCsvUpload(e, viewerTable);
        });
    }

    // =========================================================
    // 3. PAGE 2: MASTER BETTING SHEET (master_sheet.html)
    // =========================================================
    if ($('#bettingTable').length) {
        var bettingTable = $('#bettingTable').DataTable({
            scrollX: true,
            pageLength: 50,
            columns: [
                { data: 'Player' },
                { data: 'Team' },
                { data: 'Loc' },
                { 
                    data: 'Stat',
                    render: function(data) { return `<b>${data}</b>`; }
                },
                { 
                    data: 'Prediction',
                    render: function(data) { return `<span style="color:#007bff; font-weight:bold;">${data}</span>`; }
                },
                { data: 'Opp_Avg' },
                { data: 'Mins' },
                { data: 'Usg%' },
                { data: 'Pace' },
                { 
                    data: 'DvP',
                    render: function(data) {
                        if (data >= 1.1) return `<span style="color:green; font-weight:bold">${data}</span>`;
                        if (data <= 0.9) return `<span style="color:red; font-weight:bold">${data}</span>`;
                        return data;
                    }
                },
                { 
                    data: 'Trust',
                    render: function(data) {
                        let color = 'gray';
                        if (data.includes('ELITE')) color = '#28a745';
                        if (data.includes('HIGH')) color = '#ffc107';
                        if (data.includes('MEDIUM')) color = '#fd7e14';
                        if (data.includes('VOLATILE')) color = '#dc3545';
                        return `<span style="background:${color}; color:white; padding:3px 8px; border-radius:12px; font-size:0.85em;">${data}</span>`;
                    }
                },
                { data: 'Line_Over' },
                { data: 'Line_Under' }
            ],
            order: [[10, 'asc'], [4, 'desc']]
        });

        // Specific upload handler for Master Sheet page
        $('#csvFileInput').change(function(e) {
            handleCsvUpload(e, bettingTable);
        });
    }

    // =========================================================
    // SHARED HELPER FUNCTION
    // =========================================================
    function handleCsvUpload(e, tableInstance) {
        var file = e.target.files[0];
        if (!file) return;

        Papa.parse(file, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: function(results) {
                tableInstance.clear();
                if (results.data.length === 0) {
                    alert("CSV is empty or invalid.");
                    return;
                }
                tableInstance.rows.add(results.data).draw();
            },
            error: function(error) {
                console.error("Error parsing CSV:", error);
                alert("Failed to load CSV file.");
            }
        });
    }

});