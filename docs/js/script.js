$(document).ready(function() {
    
    // --- 1. DATATABLE INITIALIZATION (Existing) ---
    if ($('#predictionsTable').length) {
        var table = $('#predictionsTable').DataTable({
            scrollX: true,
            pageLength: 25,
            columns: [
                { data: 'PLAYER_NAME' },
                { data: 'TEAM_NAME' },
                { 
                    data: 'IS_HOME',
                    render: function(data) { return (data === 'True' || data === true) ? '🏠 Home' : '✈️ Away'; }
                },
                { data: 'MINUTES' },
                { data: 'USAGE_RATE' },
                { data: 'PACE' },
                { data: 'OPP_DvP' },
                { data: 'PTS' },
                { data: 'REB' },
                { data: 'AST' },
                { data: 'STL' },
                { data: 'BLK' },
                { data: 'PRA' }
            ],
            order: [[7, 'desc']]
        });
        
        // CSV Upload Logic
        $('#csvFileInput').change(function(e) {
            var file = e.target.files[0];
            if (!file) return;
            Papa.parse(file, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: function(results) {
                    table.clear();
                    if (results.data.length > 0) table.rows.add(results.data).draw();
                }
            });
        });
    }

    // --- 2. GAME PREDICTIONS JSON LOGIC (New) ---
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
            // Determine styles based on winner
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
    
    // --- 3. BETTING SHEET LOGIC (Existing, kept for safety) ---
    if ($('#bettingTable').length) {
        // ... (Keep your existing logic for master_sheet.html here) ...
    }
});