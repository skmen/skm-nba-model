$(document).ready(function() {
    // Logic for Prediction Viewer (index.html)
    if ($('#predictionsTable').length) {
        let table = $('#predictionsTable').DataTable({
            columns: [
                { data: 'PLAYER_NAME' },
                { data: 'TEAM_NAME' },
                { data: 'IS_HOME' },
                { data: 'PTS' },
                { data: 'REB' },
                { data: 'AST' },
                { data: 'STL' },
                { data: 'BLK' },
                { data: 'PREDICTION_TIME' }
            ]
        });

        $('#csvFileInput').on('change', function(e) {
            let file = e.target.files[0];
            if (file) {
                Papa.parse(file, {
                    header: true,
                    dynamicTyping: true,
                    trimHeaders: true,
                    skipEmptyLines: true,
                    complete: function(results) {
                        table.clear();
                        table.rows.add(results.data).draw();
                    }
                });
            }
        });

        let searchTimeout = null;
        $('#predictionsTable thead th').each(function() {
            var title = $(this).text();
            $(this).html(title + '<br><input type="text" class="filter-input" placeholder="Filter ' + title + '"/>');
        });

        table.columns().every(function() {
            var that = this;
            $('input', this.header()).on('keyup change clear', function() {
                var column = this;
                clearTimeout(searchTimeout);
                searchTimeout = setTimeout(function() {
                    if (that.search() !== column.value) {
                        that.search(column.value, true, false).draw();
                    }
                }, 500);
            });
        });
    }

    // Logic for Master Betting Sheet (master_sheet.html)
    if ($('#bettingTable').length) {
        var table = $("#bettingTable").DataTable({
            "pageLength": 25,
            "order": [[ 5, "asc" ], [ 4, "desc" ]], // Trust then Prediction
            "columnDefs": [
                { "className": "dt-center", "targets": "_all" },
                { "orderable": false, "targets": [3, 9] }
            ]
        });

        document.getElementById('csvFileInput').addEventListener('change', function(e) {
            var file = e.target.files[0];
            if (!file) return;

            var reader = new FileReader();
            reader.onload = function(e) {
                try {
                    var csv = e.target.result;
                    var data = parseCSV(csv);
                    if (data.length === 0) {
                        alert("Error: No valid rows found in CSV.");
                        return;
                    }
                    populateTable(data, table);
                } catch (err) {
                    alert("Error parsing CSV: " + err.message);
                }
            };
            reader.readAsText(file);
        });

        $('#bettingTable tbody').on('input', '.vegas-input', function() {
            var input = $(this);
            var prediction = parseFloat(input.data('pred'));
            var mae = parseFloat(input.data('mae'));
            var vegasLine = parseFloat(input.val());
            var statusCell = input.closest('tr').find('.status-cell');

            if (isNaN(vegasLine)) {
                statusCell.html('<span class="status-badge waiting">-</span>');
                return;
            }

            var result = calculateBetStatus(prediction, mae, vegasLine);
            statusCell.html(result);
        });

        function calculateBetStatus(prediction, mae, line) {
            var edge = prediction - line;
            var absEdge = Math.abs(edge);
            var direction = edge > 0 ? "OVER" : "UNDER";
            var strongThresh = mae;
            var leanThresh = mae * 0.8;

            if (absEdge > strongThresh) return `<span class="status-badge bet">✅ BET ${direction}</span>`;
            if (absEdge > leanThresh) return `<span class="status-badge lean">⚠️ LEAN ${direction}</span>`;
            return `<span class="status-badge pass">❌ PASS</span>`;
        }

        function parseCSV(csv) {
            var lines = csv.split(/\r\n|\n/);
            var result = [];
            var headers = lines[0].split(",");
            if (headers.length < 5) throw new Error("Invalid CSV headers.");

            for (var i = 1; i < lines.length; i++) {
                if (!lines[i]) continue;
                var currentline = lines[i].split(",");
                if (currentline.length === headers.length) {
                    var obj = {};
                    for (var j = 0; j < headers.length; j++) {
                        obj[headers[j].trim()] = currentline[j].trim();
                    }
                    result.push(obj);
                }
            }
            return result;
        }

        function populateTable(data, table) {
            table.clear();
            data.forEach(function(row) {
                var trustText = row['Trust'] || "VOLATILE";
                var trustClass = "volatile";
                if (trustText.includes("ELITE")) trustClass = "elite";
                else if (trustText.includes("HIGH")) trustClass = "high";
                else if (trustText.includes("MEDIUM")) trustClass = "medium";

                var inputField = `<input type="number" step="0.5" class="vegas-input" 
                                    data-pred="${row['Prediction']}" 
                                    data-mae="${row['MAE']}" placeholder="-">`;

                table.row.add([
                    row['Player'],
                    row['Team'],
                    `<span class="stat-badge">${row['Stat']}</span>`,
                    inputField,
                    row['Prediction'],
                    `<span class="trust-badge ${trustClass}">${trustText}</span>`,
                    row['MAE'],
                    `<span class="target-val">${row['Target_Line_Over']}</span>`,
                    `<span class="target-val">${row['Target_Line_Under']}</span>`,
                    `<span class="status-cell"><span class="status-badge waiting">-</span></span>`
                ]).draw(false);
            });
        }
    }
});
