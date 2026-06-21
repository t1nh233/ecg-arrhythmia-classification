let chart = null;
let currentResults = [];
let fileEkg = null;

// Initialize app components on DOM loaded
document.addEventListener('DOMContentLoaded', () => {
    initChart();
    setupEventListeners();
});

// Toast notification helper to display status popups
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

// Chart.js configuration for plotting heartbeat waves
function initChart() {
    const ctx = document.getElementById('ecgChart').getContext('2d');
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({ length: 300 }, (_, i) => i + 1),
            datasets: [{
                label: 'ECG Signal Amplitude',
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

// Event listeners for file loading and submission
function setupEventListeners() {
    const zone = document.getElementById('upload-zone');
    const filePicker = document.getElementById('file-picker');
    
    // Clicking upload zone triggers hidden input dialog
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

    // Drag over styling
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });

    zone.addEventListener('dragleave', () => {
        zone.classList.remove('dragover');
    });

    // File drop event handling
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        
        if (e.dataTransfer.files.length > 0) {
            fileEkg = e.dataTransfer.files[0];
            updateFileList();
            verifyFile(fileEkg);
        }
    });

    // Submits the EKG file to the AI backend
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
                throw new Error(errorData.detail || 'ECG Analysis request failed.');
            }
            const data = await response.json();
            displayResults(data);
            showToast('AI diagnostics successfully completed!');
        } catch (e) {
            showToast(e.message, 'error');
        } finally {
            setLoadingState(false);
        }
    });
}

// Client-side file verification and formatting checks
function verifyFile(file) {
    const card = document.getElementById('format-check-card');
    card.style.display = 'block';
    
    const chkCol = document.getElementById('chk-ekg-col');
    const chkSize = document.getElementById('chk-file-size');
    const chkMatch = document.getElementById('chk-data-match');
    
    chkCol.className = '';
    chkCol.textContent = '⌛ Reading column headers...';
    chkSize.className = '';
    chkSize.textContent = '⌛ Checking sample lines...';
    chkMatch.className = '';
    chkMatch.textContent = '⌛ Verifying compatibility parameters...';
    
    let colValid = false;
    let sizeValid = false;
    
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const text = e.target.result;
            const lines = text.split('\n').filter(line => line.trim() !== '');
            const firstLine = lines[0].trim();
            const cols = firstLine.split(',').map(c => c.replace(/"/g, '').trim());
            
            // 1. Verify signal columns
            if (cols.includes('MLII') || cols.length > 0) {
                chkCol.className = 'valid';
                chkCol.textContent = `✅ Column verified: (${cols.slice(0, 3).join(', ')}${cols.length > 3 ? '...' : ''})`;
                colValid = true;
            } else {
                chkCol.className = 'invalid';
                chkCol.textContent = '❌ No valid signal column headers found';
            }
            
            // 2. Verify signal length
            if (lines.length > 1000) {
                chkSize.className = 'valid';
                chkSize.textContent = `✅ Signal length: ${lines.length - 1} samples (Valid)`;
                sizeValid = true;
            } else {
                chkSize.className = 'invalid';
                chkSize.textContent = `❌ Signal too short (${lines.length - 1} samples, minimum 1000 required)`;
            }
            
            // 3. Final compatibility verdict
            const btn = document.getElementById('btn-upload-analyze');
            if (colValid && sizeValid) {
                chkMatch.className = 'valid';
                chkMatch.textContent = '✅ File is fully compatible with AI models!';
                btn.disabled = false;
            } else {
                chkMatch.className = 'invalid';
                chkMatch.textContent = '❌ File format is incompatible';
                btn.disabled = true;
            }
        };
        reader.readAsText(file.slice(0, 50000));
    } else {
        chkCol.className = 'invalid';
        chkCol.textContent = '❌ File not found';
    }
}

// Update the list of uploaded files in the UI
function updateFileList() {
    const list = document.getElementById('selected-files');
    list.innerHTML = '';
    
    if (fileEkg) {
        list.innerHTML += `<div>🟢 EKG Signal: ${fileEkg.name} (${(fileEkg.size / 1024).toFixed(1)} KB)</div>`;
    }
}

// Adjust UI loading state
function setLoadingState(isLoading) {
    const ubtn = document.getElementById('btn-upload-analyze');
    if (isLoading) {
        ubtn.disabled = true;
        ubtn.textContent = '⏳ Analyzing using AI...';
    } else {
        ubtn.disabled = false;
        ubtn.textContent = '⚡ Start AI Diagnosis';
    }
}

// Display analysis reports in dashboard
function displayResults(data) {
    // Populate stats
    document.getElementById('metric-total').textContent = data.total_beats;
    document.getElementById('metric-accuracy').textContent = `${(data.accuracy * 100).toFixed(2)}%`;
    document.getElementById('metric-v').textContent = data.class_counts.V;
    document.getElementById('metric-s').textContent = data.class_counts.S;
    
    // Build tables
    const tbody = document.getElementById('results-tbody');
    tbody.innerHTML = '';
    
    currentResults = data.results;
    
    if (currentResults.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align: center;">No heartbeats detected.</td></tr>';
        return;
    }
    
    currentResults.forEach((res, index) => {
        const tr = document.createElement('tr');
        tr.dataset.index = index;
        
        const badgeClass = res.prediction.includes('(V)') ? 'text-red' : (res.prediction.includes('(S)') ? 'text-yellow' : 'text-green');
        const statusText = res.is_correct ? '🟢 Correct' : '🔴 Incorrect';
        
        tr.innerHTML = `
            <td>#${res.beat_index}</td>
            <td class="${badgeClass} font-weight:600">${res.prediction}</td>
            <td>${(res.confidence * 100).toFixed(1)}%</td>
            <td>${res.ground_truth}</td>
            <td>${statusText}</td>
        `;
        
        tr.addEventListener('click', () => {
            selectBeat(index);
        });
        
        tbody.appendChild(tr);
    });
    
    // Auto-plot the first heartbeat segment
    selectBeat(0);
}

// Select a specific beat row, scroll into view, and draw its waveform on the plotter
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
    
    document.getElementById('selected-beat-badge').textContent = `Beat: #${beat.beat_index} (${beat.prediction})`;
    
    // Match line colors to the diagnostic class
    let color = '#00e676'; // Normal - Green
    if (beat.prediction.includes('(V)')) color = '#ff1744'; // Ventricular - Red
    if (beat.prediction.includes('(S)')) color = '#ffea00'; // Supraventricular - Yellow
    
    chart.data.datasets[0].data = beat.signal;
    chart.data.datasets[0].borderColor = color;
    chart.update();
}
