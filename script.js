$(document).ready(function() {
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

    // Setup a timer for regex search
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
            }, 500); // 500ms delay
        });
    });
});
