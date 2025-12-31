// Elements
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const selectedFile = document.getElementById('selectedFile');
const uploadStatus = document.getElementById('uploadStatus');
const questionInput = document.getElementById('questionInput');
const askBtn = document.getElementById('askBtn');
const response = document.getElementById('response');
const sourcesDiv = document.getElementById('sources');
const fileList = document.getElementById('fileList');
const fileCount = document.getElementById('fileCount');
const historyList = document.getElementById('historyList');

// Load initial data
loadFiles();
loadHistory();

// File selection
fileInput.addEventListener('change', function () {
    if (this.files.length > 0) {
        uploadBtn.disabled = false;
        selectedFile.textContent = `📄 Selected: ${this.files[0].name}`;
    } else {
        uploadBtn.disabled = true;
        selectedFile.textContent = '';
    }
});

// Enter key support
questionInput.addEventListener('keypress', function (e) {
    if (e.key === 'Enter' && !askBtn.disabled) {
        askQuestion();
    }
});

// Tab switching
function switchTab(tab) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

    event.target.classList.add('active');
    document.getElementById(tab + 'Tab').classList.add('active');

    if (tab === 'history') {
        loadHistory();
    }
}

// Upload file
async function uploadFile() {
    const file = fileInput.files[0];
    if (!file) {
        showStatus('Please select a file first', 'error');
        return;
    }

    uploadBtn.disabled = true;
    uploadBtn.innerHTML = '<span class="loading"></span>';
    uploadStatus.style.display = 'none';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await res.json();

        if (res.ok) {
            showStatus(data.message, 'success');
            fileInput.value = '';
            selectedFile.textContent = '';
            loadFiles();

            // Enable question input
            questionInput.disabled = false;
            askBtn.disabled = false;
            questionInput.focus();
        } else {
            showStatus(data.message || 'Upload failed', 'error');
        }
    } catch (error) {
        showStatus(`Error: ${error.message}`, 'error');
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = '⬆️ Upload';
    }
}

// Ask question
async function askQuestion() {
    const question = questionInput.value.trim();
    if (!question) {
        alert('Please enter a question');
        return;
    }

    askBtn.disabled = true;
    askBtn.innerHTML = '<span class="loading"></span>';
    response.textContent = '🤔 Thinking...';
    sourcesDiv.style.display = 'none';

    try {
        const res = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: question })
        });

        const data = await res.json();

        if (res.ok) {
            response.textContent = data.answer;

            if (data.sources && data.sources.length > 0) {
                sourcesDiv.innerHTML = `<strong>📚 Sources:</strong> ${data.sources.map(s => s.split('/').pop()).join(', ')}`;
                sourcesDiv.style.display = 'block';
            }

            // Clear question input
            questionInput.value = '';

            // Reload history if on history tab
            if (document.getElementById('historyTab').classList.contains('active')) {
                loadHistory();
            }
        } else {
            response.textContent = data.message || 'Failed to get answer';
        }
    } catch (error) {
        response.textContent = `Error: ${error.message}`;
    } finally {
        askBtn.disabled = false;
        askBtn.textContent = 'Ask';
    }
}

// Load files
async function loadFiles() {
    try {
        const res = await fetch('/files');
        const data = await res.json();

        if (data.files && data.files.length > 0) {
            fileCount.textContent = data.files.length;
            fileList.innerHTML = data.files.map(file => `
                        <div class="file-item">
                            <div class="file-info">
                                <div class="file-name">📄 ${file.filename}</div>
                                <div class="file-meta">${file.size_mb} MB • ${file.chunks} chunks</div>
                            </div>
                            <button class="delete-btn" onclick="deleteFile('${file.filename}')">🗑️</button>
                        </div>
                    `).join('');
        } else {
            fileCount.textContent = '0';
            fileList.innerHTML = '<div class="empty-state">No files uploaded yet</div>';
        }
    } catch (error) {
        console.error('Error loading files:', error);
    }
}

// Delete file
async function deleteFile(filename) {
    if (!confirm(`Delete "${filename}"?`)) return;

    try {
        const res = await fetch(`/files/${filename}`, {
            method: 'DELETE'
        });

        const data = await res.json();

        if (res.ok) {
            showStatus(data.message, 'success');
            loadFiles();

            // Disable question input if no files left
            if (data.remaining_files === 0) {
                questionInput.disabled = true;
                askBtn.disabled = true;
                response.textContent = '';
                sourcesDiv.style.display = 'none';
            }
        } else {
            showStatus(data.message || 'Delete failed', 'error');
        }
    } catch (error) {
        showStatus(`Error: ${error.message}`, 'error');
    }
}

// Load history
async function loadHistory() {
    try {
        const res = await fetch('/history');
        const data = await res.json();

        if (data.history && data.history.length > 0) {
            historyList.innerHTML = data.history.reverse().map(item => {
                const date = new Date(item.timestamp).toLocaleString();
                return `
                            <div class="history-item">
                                <div class="history-question">Q: ${item.question}</div>
                                <div class="history-answer">A: ${item.answer}</div>
                                <div class="history-meta">🕒 ${date}</div>
                            </div>
                        `;
            }).join('');
        } else {
            historyList.innerHTML = '<div class="empty-state">No conversation history yet</div>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

// Clear history
async function clearHistory() {
    if (!confirm('Clear all conversation history?')) return;

    try {
        const res = await fetch('/history', {
            method: 'DELETE'
        });

        if (res.ok) {
            loadHistory();
            alert('History cleared successfully');
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

// Show status message
function showStatus(message, type) {
    uploadStatus.className = `status ${type}`;
    uploadStatus.textContent = message;
    uploadStatus.style.display = 'block';

    setTimeout(() => {
        uploadStatus.style.display = 'none';
    }, 5000);
}