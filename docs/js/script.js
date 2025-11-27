$(document).ready(function() {

    // =========================================================
    // 1. GAME PREDICTIONS JSON LOGIC (Scoreboard)
    // =========================================================
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
                { data: 'PLAYER_NAME' },
                { data: 'TEAM_NAME' },
                { 
                    data: 'IS_HOME',
                    defaultContent: "",
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
            order: [[7, 'desc']] 
        });

        // Use the shared helper function
        setupCsvUpload(viewerTable, '#csvFileInput');
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
                    render: function(data) { return `<span class="mae-val" style="color:#999; font-size:0.9em;">${data}</span>`; }
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
                // --- Interactive Columns ---
                {
                    data: null, 
                    orderable: false,
                    render: function() { return `<input type="number" step="0.5" class="vegas-input" placeholder="-">`; }
                },
                {
                    data: null, 
                    orderable: false,
                    className: "rec-cell",
                    render: function() { return `<span class="rec-placeholder" style="color:#ccc;">-</span>`; }
                },
                // ---------------------------
                { 
                    data: 'Trust',
                    render: function(data) {
                        let color = '#6c757d';
                        if (data && data.includes('ELITE')) color = '#28a745';
                        if (data && data.includes('HIGH')) color = '#ffc107';
                        if (data && data.includes('MEDIUM')) color = '#fd7e14';
                        if (data && data.includes('VOLATILE')) color = '#dc3545';
                        return `<span style="background:${color}; color:white; padding:3px 8px; border-radius:12px; font-size:0.85em;">${data}</span>`;
                    }
                },
                { data: 'Line_Over' },
                { data: 'Line_Under' }
            ],
            order: [[4, 'desc']] 
        });

        // Use the shared helper function
        setupCsvUpload(bettingTable, '#csvFileInput');

        // --- REAL-TIME CALCULATOR LOGIC ---
        $('#bettingTable tbody').on('keyup change', '.vegas-input', function() {
            var $row = $(this).closest('tr');
            var vegasLine = parseFloat($(this).val());
            
            // Get row data from DataTables
            var rowData = bettingTable.row($row).data();
            var prediction = parseFloat(rowData.Prediction);
            var mae = parseFloat(rowData.MAE); 

            var $recCell = $row.find('.rec-cell');

            if (isNaN(vegasLine) || isNaN(mae)) {
                $recCell.html('<span class="rec-placeholder" style="color:#ccc;">-</span>');
                return;
            }

            var edge = prediction - vegasLine;
            var absEdge = Math.abs(edge);
            var direction = edge > 0 ? "OVER" : "UNDER";
            
            // Logic: Strong > 100% MAE, Lean > 80% MAE
            if (absEdge > mae) {
                $recCell.html(`<span class="rec-badge rec-bet" style="background:#28a745; color:white; padding:4px 8px; border-radius:4px; font-weight:bold;">BET ${direction}</span>`);
            } else if (absEdge > (mae * 0.8)) {
                $recCell.html(`<span class="rec-badge rec-lean" style="background:#ffc107; color:black; padding:4px 8px; border-radius:4px; font-weight:bold;">LEAN ${direction}</span>`);
            } else {
                $recCell.html(`<span class="rec-badge rec-pass" style="background:#dc3545; color:white; padding:4px 8px; border-radius:4px; font-weight:bold;">PASS</span>`);
            }
        });
    }

    // =========================================================
    // SHARED HELPER FUNCTION
    // =========================================================
    function setupCsvUpload(tableInstance, inputSelector) {
        // Remove any previous event handlers to prevent duplicates
        $(inputSelector).off('change').on('change', function(e) {
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
        });
    }

});