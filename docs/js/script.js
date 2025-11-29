$(document).ready(function() {

    // =========================================================
    // 1. GAME PREDICTIONS & CALCULATOR LOGIC
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

        games.forEach(function(game, index) {
            var awayClass = game.winner === game.away_team ? 'winner' : '';
            var homeClass = game.winner === game.home_team ? 'winner' : '';
            
            // Generate unique IDs for this game's inputs
            var id = index; 

            var cardHtml = `
                <div class="score-card" data-away="${game.away_team}" data-home="${game.home_team}" 
                     data-away-score="${game.away_score}" data-home-score="${game.home_score}"
                     data-total="${game.total}">
                    
                    <h3>${game.away_team} @ ${game.home_team}</h3>
                    
                    <div class="score-row ${awayClass}">
                        <span>${game.away_team}</span><span>${game.away_score}</span>
                    </div>
                    <div class="score-row ${homeClass}">
                        <span>${game.home_team}</span><span>${game.home_score}</span>
                    </div>
                    <div style="text-align:center; font-size:0.85em; color:#666; margin-top:5px;">
                        Model Spread: ${game.winner} -${game.spread} | Total: ${game.total}
                    </div>

                    <div class="bet-section">
                        <div class="bet-section-title">Moneyline (Odds)</div>
                        <div class="input-grid">
                            <span class="team-label">${game.away_team}</span>
                            <input type="number" class="calc-input ml-input" data-team="away" placeholder="-110">
                            <div class="rec-box rec-empty ml-rec" rowspan="2">PASS</div>
                        </div>
                        <div class="input-grid" style="margin-top:4px;">
                            <span class="team-label">${game.home_team}</span>
                            <input type="number" class="calc-input ml-input" data-team="home" placeholder="+110">
                        </div>
                    </div>

                    <div class="bet-section">
                        <div class="bet-section-title">Spread (Line / Odds)</div>
                        <div class="input-grid">
                            <span class="team-label">${game.away_team}</span>
                            <div class="input-group">
                                <input type="number" class="calc-input spread-line" data-team="away" placeholder="+5.5">
                                <input type="number" class="calc-input spread-odds" placeholder="-110">
                            </div>
                            <div class="rec-box rec-empty spread-rec">PASS</div>
                        </div>
                        <div class="input-grid" style="margin-top:4px;">
                            <span class="team-label">${game.home_team}</span>
                            <div class="input-group">
                                <input type="number" class="calc-input spread-line" data-team="home" placeholder="-5.5">
                                <input type="number" class="calc-input spread-odds" placeholder="-110">
                            </div>
                        </div>
                    </div>

                    <div class="bet-section">
                        <div class="bet-section-title">Total Points (Line / Odds)</div>
                        <div class="input-grid">
                            <span class="team-label">Over</span>
                            <div class="input-group">
                                <input type="number" class="calc-input total-line" placeholder="210.5">
                                <input type="number" class="calc-input total-odds" placeholder="-110">
                            </div>
                            <div class="rec-box rec-empty total-rec">PASS</div>
                        </div>
                        <div class="input-grid" style="margin-top:4px;">
                            <span class="team-label">Under</span>
                            <div class="input-group">
                                <input type="number" class="calc-input total-line-dummy" placeholder="-" disabled style="background:#eee;">
                                <input type="number" class="calc-input total-odds" placeholder="-110">
                            </div>
                        </div>
                    </div>

                </div>
            `;
            container.append(cardHtml);
        });

        attachCalculatorEvents();
    }

    function attachCalculatorEvents() {
        // --- MONEYLINE LOGIC ---
        $('.ml-input').on('keyup change', function() {
            var $card = $(this).closest('.score-card');
            var awayName = $card.data('away');
            var homeName = $card.data('home');
            var awayOdds = parseFloat($card.find('.ml-input[data-team="away"]').val());
            var homeOdds = parseFloat($card.find('.ml-input[data-team="home"]').val());
            var awayScore = parseFloat($card.data('away-score'));
            var homeScore = parseFloat($card.data('home-score'));
            var margin = Math.abs(awayScore - homeScore);
            var modelWinner = awayScore > homeScore ? 'away' : 'home';
            var targetTeamName = modelWinner === 'away' ? awayName : homeName;
            var $recBox = $card.find('.ml-rec');
            var relevantOdds = modelWinner === 'away' ? awayOdds : homeOdds;

            if (isNaN(relevantOdds)) { updateBadge($recBox, "PASS", "rec-empty"); return; }

            var rec = "PASS";
            var style = "rec-pass";
            if (margin > 10 && relevantOdds > -250) {
                rec = `STRONG<br><span style="font-size:0.9em">${targetTeamName}</span>`;
                style = "rec-strong";
            } else if (margin > 5 && relevantOdds > -180) {
                rec = `LEAN<br><span style="font-size:0.9em">${targetTeamName}</span>`;
                style = "rec-lean";
            }
            updateBadge($recBox, rec, style);
        });

        // --- SPREAD LOGIC ---
        $('.spread-line').on('keyup change', function() {
            var $card = $(this).closest('.score-card');
            var line = parseFloat($(this).val());
            var teamType = $(this).data('team');
            var teamName = (teamType === 'home') ? $card.data('home') : $card.data('away');
            var awayScore = parseFloat($card.data('away-score'));
            var homeScore = parseFloat($card.data('home-score'));
            var $recBox = $card.find('.spread-rec');

            if (isNaN(line)) { updateBadge($recBox, "PASS", "rec-empty"); return; }

            var modelMargin = (teamType === 'home') ? (homeScore - awayScore) : (awayScore - homeScore);
            var predictedScoreWithLine = modelMargin + line; 
            var rec = "PASS";
            var style = "rec-pass";

            if (predictedScoreWithLine > 4.5) {
                rec = `STRONG<br><span style="font-size:0.9em">${teamName}</span>`;
                style = "rec-strong";
            } else if (predictedScoreWithLine > 2.5) {
                rec = `LEAN<br><span style="font-size:0.9em">${teamName}</span>`;
                style = "rec-lean";
            }
            updateBadge($recBox, rec, style);
        });

        // --- TOTAL LOGIC ---
        $('.total-line').on('keyup change', function() {
            var $card = $(this).closest('.score-card');
            var vegasTotal = parseFloat($(this).val());
            var modelTotal = parseFloat($card.data('total'));
            var $recBox = $card.find('.total-rec');

            if (isNaN(vegasTotal)) { updateBadge($recBox, "PASS", "rec-empty"); return; }

            var diff = modelTotal - vegasTotal;
            var absDiff = Math.abs(diff);
            var direction = diff > 0 ? "OVER" : "UNDER";
            var rec = "PASS";
            var style = "rec-pass";

            if (absDiff > 6.0) {
                rec = `STRONG ${direction}`;
                style = "rec-strong";
            } else if (absDiff > 3.0) {
                rec = `LEAN ${direction}`;
                style = "rec-lean";
            }
            updateBadge($recBox, rec, style);
        });
    }

    function updateBadge($el, htmlContent, styleClass) {
        $el.html(htmlContent)
           .removeClass("rec-strong rec-lean rec-pass rec-empty")
           .addClass(styleClass);
    }

    // =========================================================
    // 2. PAGE 1: PREDICTIONS VIEWER TABLE (Existing)
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
                    render: function(data) { return (data === 'True' || data === true) ? '🏠 Home' : '✈️ Away'; }
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
        setupCsvUpload(viewerTable, '#csvFileInput');
    }

    // =========================================================
    // 3. PAGE 2: MASTER BETTING SHEET TABLE (Existing)
    // =========================================================
    if ($('#bettingTable').length) {
        var bettingTable = $('#bettingTable').DataTable({
            scrollX: true,
            pageLength: 50,
            columns: [
                { data: 'Player' },
                { data: 'Team' },
                { data: 'Loc' },
                { data: 'Stat', render: function(d) { return `<b>${d}</b>`; } },
                { data: 'Prediction', render: function(d) { return `<span class="pred-val" style="color:#007bff; font-weight:bold;">${d}</span>`; } },
                { data: 'MAE', render: function(d) { return `<span class="mae-val" style="color:#999; font-size:0.9em;">${d}</span>`; } },
                { data: 'Opp_Avg' },
                { data: 'Mins' },
                { data: 'Usg%' },
                { data: 'Pace' },
                { data: 'DvP', render: function(d) { return d >= 1.1 ? `<span style="color:green;font-weight:bold">${d}</span>` : d <= 0.9 ? `<span style="color:red;font-weight:bold">${d}</span>` : d; } },
                { data: null, orderable: false, render: function() { return `<input type="number" step="0.5" class="vegas-input" placeholder="-">`; } },
                { data: null, orderable: false, className: "rec-cell", render: function() { return `<span class="rec-placeholder" style="color:#ccc;">-</span>`; } },
                { data: 'Trust', render: function(d) { return `<span style="background:${d.includes('ELITE')?'#28a745':d.includes('HIGH')?'#ffc107':'#dc3545'}; color:white; padding:3px 8px; border-radius:12px;">${d}</span>`; } },
                { data: 'Line_Over' },
                { data: 'Line_Under' }
            ],
            order: [[4, 'desc']] 
        });

        setupCsvUpload(bettingTable, '#csvFileInput');

        $('#bettingTable tbody').on('keyup change', '.vegas-input', function() {
            var $row = $(this).closest('tr');
            var vegasLine = parseFloat($(this).val());
            var rowData = bettingTable.row($row).data();
            var prediction = parseFloat(rowData.Prediction);
            var mae = parseFloat(rowData.MAE);
            var $recCell = $row.find('.rec-cell');

            if (isNaN(vegasLine) || isNaN(mae)) { $recCell.html('-'); return; }

            var edge = prediction - vegasLine;
            var absEdge = Math.abs(edge);
            var dir = edge > 0 ? "OVER" : "UNDER";
            
            if (absEdge > mae) {
                $recCell.html(`<span class="rec-badge rec-strong" style="background:#28a745; color:white; padding:4px;">BET ${dir}</span>`);
            } else if (absEdge > (mae * 0.8)) {
                $recCell.html(`<span class="rec-badge rec-lean" style="background:#ffc107; color:black; padding:4px;">LEAN ${dir}</span>`);
            } else {
                $recCell.html(`<span class="rec-badge rec-pass" style="background:#dc3545; color:white; padding:4px;">PASS</span>`);
            }
        });
    }

    // =========================================================
    // 4. PAGE 3: SIMULATION VIEWER TABLE (New)
    // =========================================================
    if ($('#simulationTable').length) {
        var simTable = $('#simulationTable').DataTable({
            scrollX: true,
            pageLength: 25,
            columns: [
                { data: 'PLAYER_NAME' },
                { data: 'TEAM_NAME' },
                { data: 'MINUTES', render: $.fn.dataTable.render.number(',', '.', 1) },
                { 
                    data: 'PTS', 
                    render: function(data) { 
                        return `<span class="pts-val">${parseFloat(data).toFixed(1)}</span>`; 
                    } 
                },
                { data: 'REB', render: $.fn.dataTable.render.number(',', '.', 1) },
                { data: 'AST', render: $.fn.dataTable.render.number(',', '.', 1) },
                { data: 'STL', render: $.fn.dataTable.render.number(',', '.', 1) },
                { data: 'BLK', render: $.fn.dataTable.render.number(',', '.', 1) },
                { data: 'FG3M', render: $.fn.dataTable.render.number(',', '.', 1) },
                { 
                    data: 'USAGE_PCT', 
                    render: function(data) {
                        // Assuming USAGE_PCT is decimal (e.g. 0.302)
                        return `<span class="usage-val">${(parseFloat(data) * 100).toFixed(1)}%</span>`;
                    }
                }
            ],
            order: [[3, 'desc']] // Sort by PTS descending by default
        });

        setupCsvUpload(simTable, '#simFileInput');
    }

    // =========================================================
    // SHARED CSV UPLOAD HELPER
    // =========================================================
    function setupCsvUpload(tableInstance, inputSelector) {
        $(inputSelector).off('change').on('change', function(e) {
            var file = e.target.files[0];
            if (!file) return;
            Papa.parse(file, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: function(results) {
                    tableInstance.clear();
                    if (results.data.length > 0) tableInstance.rows.add(results.data).draw();
                },
                error: function(error) { alert("Failed to load CSV file."); }
            });
        });
    }
});