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
                    render: function(data) { return `<span class="pred-val" style="color:#007bff; font-weight:bold;">${data}</span>`; }
                },
                { 
                    data: 'MAE',
                    render: function(data) { return `<span class="mae-val" style="color:#666;">${data}</span>`; }
                },
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
                // --- INTERACTIVE COLUMNS ---
                {
                    data: null, // Input Column
                    orderable: false,
                    render: function(data, type, row) {
                        return `<input type="number" step="0.5" class="vegas-input" placeholder="-">`;
                    }
                },
                {
                    data: null, // Result Column
                    orderable: false,
                    className: "rec-cell",
                    render: function(data, type, row) {
                        return `<span class="rec-placeholder" style="color:#ccc;">-</span>`;
                    }
                },
                // ---------------------------
                { data: 'Line_Over' },
                { data: 'Line_Under' }
            ],
            order: [[4, 'desc']] // Sort by Prediction High-to-Low
        });

        // --- REAL-TIME CALCULATION LOGIC ---
        $('#bettingTable tbody').on('keyup change', '.vegas-input', function() {
            var $row = $(this).closest('tr');
            var vegasLine = parseFloat($(this).val());
            
            // Get data from the DataTable row 
            // (We use data() so it works even if columns are hidden/responsive)
            var rowData = bettingTable.row($row).data();
            var prediction = parseFloat(rowData.Prediction);
            var mae = parseFloat(rowData.MAE);

            var $recCell = $row.find('.rec-cell');

            if (isNaN(vegasLine)) {
                $recCell.html('<span class="rec-placeholder" style="color:#ccc;">-</span>');
                return;
            }

            // --- THE BETTING LOGIC (Matches bet_analyzer.py) ---
            var edge = prediction - vegasLine;
            var absEdge = Math.abs(edge);
            var direction = edge > 0 ? "OVER" : "UNDER";
            
            // Thresholds
            var strongThreshold = mae * 1.0; // 100% of MAE
            var leanThreshold = mae * 0.8;   // 80% of MAE

            var html = "";

            if (absEdge > strongThreshold) {
                // STRONG BET (Green)
                html = `<span class="rec-badge rec-bet">BET ${direction}</span>`;
            } else if (absEdge > leanThreshold) {
                // LEAN (Yellow)
                html = `<span class="rec-badge rec-lean">LEAN ${direction}</span>`;
            } else {
                // PASS (Red)
                html = `<span class="rec-badge rec-pass">PASS</span>`;
            }

            // Optional: Show the edge value underneath
            html += `<div style="font-size:0.75em; margin-top:3px; color:#555;">Edge: ${absEdge.toFixed(1)}</div>`;

            $recCell.html(html);
        });

        setupCsvUpload(table, '#csvFileInput');
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