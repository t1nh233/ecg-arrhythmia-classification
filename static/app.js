let chart = null;
let currentResults = [];
let fileEkg = null;
let fileAnn = null;

document.addEventListener('DOMContentLoaded', () => {
    initChart();
    setupEventListeners();
});

// Toast notification helper
function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    
    if (type === 'error') {
        toast.style.borderLeftColor = '#ff1744';
        toast.style.borderColor = '#ff1744';
    } else {
        toast.style.borderLeftColor = '#00e5ff';
        toast.style.borderColor = '#00e5ff';
    }
    
    toast.classList.add('show');
    setTimeout(() => {
        toast.classList.remove('show');
    }, 4000);
}

// Chart initialization
function initChart() {
    const ctx = document.getElementById('ecgChart').getContext('2d');
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({ length: 300 }, (_, i) => i + 1),
            datasets: [{
                label: 'Tín hiệu điện tâm đồ (ECG MLII)',
                data: Array(300).fill(0),
                borderColor: '#00e5ff',
                borderWidth: 2.5,
                fill: false,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.03)' },
                    ticks: { color: '#8b9bb4', maxTicksLimit: 10 }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#8b9bb4' }
                }
            }
        }
    });
}

// Setup Event Handlers
function setupEventListeners() {
    // File Upload Zone interaction
    const zone = document.getElementById('upload-zone');
    const filePicker = document.getElementById('file-picker');
    
    zone.addEventListener('click', () => {
        filePicker.click();
    });
 
    filePicker.addEventListener('change', () => {
        if (filePicker.files.length > 0) {
            fileEkg = filePicker.files[0];
            updateFileList();
            verifyFile(fileEkg);
        }
    });

    // Drag-and-drop
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });

    zone.addEventListener('dragleave', () => {
        zone.classList.remove('dragover');
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        
        if (e.dataTransfer.files.length > 0) {
            fileEkg = e.dataTransfer.files[0];
            updateFileList();
            verifyFile(fileEkg);
        }
    });

    // Analyze Custom Uploaded files
    document.getElementById('btn-upload-analyze').addEventListener('click', async () => {
        if (!fileEkg) return;
        
        const formData = new FormData();
        formData.append('ekg_file', fileEkg);
        
        setLoadingState(true);
        try {
            const response = await fetch('/api/predict/upload', {
                method: 'POST',
                body: formData
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Lỗi phân tích tệp EKG.');
            }
            const data = await response.json();
            displayResults(data);
            showToast('Chẩn đoán thành công tệp EKG!');
        } catch (e) {
            showToast(e.message, 'error');
        } finally {
            setLoadingState(false);
        }
    });
}

function verifyFile(file) {
    const card = document.getElementById('format-check-card');
    card.style.display = 'block';
    
    const chkCol = document.getElementById('chk-ekg-col');
    const chkSize = document.getElementById('chk-file-size');
    const chkMatch = document.getElementById('chk-data-match');
    
    chkCol.className = '';
    chkCol.textContent = '⌛ Đang đọc tiêu đề cột...';
    chkSize.className = '';
    chkSize.textContent = '⌛ Đang kiểm tra số dòng...';
    chkMatch.className = '';
    chkMatch.textContent = '⌛ Đang đối chiếu tham số...';
    
    let colValid = false;
    let sizeValid = false;
    
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const text = e.target.result;
            const lines = text.split('\n').filter(line => line.trim() !== '');
            const firstLine = lines[0].trim();
            const cols = firstLine.split(',').map(c => c.replace(/"/g, '').trim());
            
            // Check signal columns
            if (cols.includes('MLII') || cols.length > 0) {
                chkCol.className = 'valid';
                chkCol.textContent = `✅ Cột dữ liệu: (${cols.slice(0, 3).join(', ')}${cols.length > 3 ? '...' : ''})`;
                colValid = true;
            } else {
                chkCol.className = 'invalid';
                chkCol.textContent = '❌ Không tìm thấy cột tín hiệu hợp lệ';
            }
            
            // Check file size / row count (needs to be long enough for heartbeat detection, e.g. > 1000 samples)
            if (lines.length > 1000) {
                chkSize.className = 'valid';
                chkSize.textContent = `✅ Độ dài tín hiệu: ${lines.length - 1} samples (Hợp lệ)`;
                sizeValid = true;
            } else {
                chkSize.className = 'invalid';
                chkSize.textContent = `❌ Tín hiệu quá ngắn (${lines.length - 1} samples, yêu cầu > 1000)`;
            }
            
            // Final validation status
            const btn = document.getElementById('btn-upload-analyze');
            if (colValid && sizeValid) {
                chkMatch.className = 'valid';
                chkMatch.textContent = '✅ Đủ điều kiện phân tích AI!';
                btn.disabled = false;
            } else {
                chkMatch.className = 'invalid';
                chkMatch.textContent = '❌ Tập tin không tương thích';
                btn.disabled = true;
            }
        };
        reader.readAsText(file.slice(0, 50000)); // Read first 50KB for checks
    } else {
        chkCol.className = 'invalid';
        chkCol.textContent = '❌ Không tìm thấy tệp';
    }
}

function updateFileList() {
    const list = document.getElementById('selected-files');
    list.innerHTML = '';
    
    if (fileEkg) {
        list.innerHTML += `<div>🟢 EKG: ${fileEkg.name} (${(fileEkg.size / 1024).toFixed(1)} KB)</div>`;
    }
}

function setLoadingState(isLoading) {
    const ubtn = document.getElementById('btn-upload-analyze');
    if (isLoading) {
        ubtn.disabled = true;
        ubtn.textContent = '⏳ Đang phân tích bằng AI...';
    } else {
        ubtn.disabled = false;
        ubtn.textContent = '⚡ Bắt đầu Phân tích AI';
    }
}

// Display results in GUI
function displayResults(data) {
    // Metrics
    document.getElementById('metric-total').textContent = data.total_beats;
    document.getElementById('metric-accuracy').textContent = `${(data.accuracy * 100).toFixed(2)}%`;
    document.getElementById('metric-v').textContent = data.class_counts.V;
    document.getElementById('metric-s').textContent = data.class_counts.S;
    
    // Table
    const tbody = document.getElementById('results-tbody');
    tbody.innerHTML = '';
    
    currentResults = data.results;
    
    if (currentResults.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align: center;">Không có nhịp tim nào</td></tr>';
        return;
    }
    
    currentResults.forEach((res, index) => {
        const tr = document.createElement('tr');
        tr.dataset.index = index;
        
        const badgeClass = res.prediction.includes('(V)') ? 'text-red' : (res.prediction.includes('(S)') ? 'text-yellow' : 'text-green');
        const correctText = res.is_correct ? '🟢 Đúng' : '🔴 Sai';
        
        tr.innerHTML = `
            <td>#${res.beat_index}</td>
            <td class="${badgeClass} font-weight:600">${res.prediction}</td>
            <td>${(res.confidence * 100).toFixed(1)}%</td>
            <td>${res.ground_truth}</td>
            <td>${correctText}</td>
        `;
        
        tr.addEventListener('click', () => {
            selectBeat(index);
        });
        
        tbody.appendChild(tr);
    });
    
    // Plot the very first beat by default
    selectBeat(0);
}

// Highlight a heartbeat and plot its signal
function selectBeat(index) {
    const rows = document.querySelectorAll('#results-tbody tr');
    rows.forEach(r => r.classList.remove('active-row'));
    
    const selectedRow = document.querySelector(`#results-tbody tr[data-index="${index}"]`);
    if (selectedRow) {
        selectedRow.classList.add('active-row');
        selectedRow.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
    
    const beat = currentResults[index];
    if (!beat) return;
    
    document.getElementById('selected-beat-badge').textContent = `Nhịp số: #${beat.beat_index} (${beat.prediction})`;
    
    // Determine color based on prediction class
    let color = '#00e676'; // Normal - Green
    if (beat.prediction.includes('(V)')) color = '#ff1744'; // Ventricular - Red
    if (beat.prediction.includes('(S)')) color = '#ffea00'; // Supraventricular - Yellow
    
    chart.data.datasets[0].data = beat.signal;
    chart.data.datasets[0].borderColor = color;
    chart.update();
}
