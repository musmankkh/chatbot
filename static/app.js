// Element references
const questionInput = document.getElementById('questionInput');
const askBtn = document.getElementById('askBtn');
const response = document.getElementById('response');
const sourcesDiv = document.getElementById('sources');
const historyList = document.getElementById('historyList');
const chatbotToggle = document.getElementById('chatbotToggle');
const chatbotContainer = document.getElementById('chatbotContainer');
const scrollIndicator = document.getElementById('scrollIndicator');

// Scroll event listener
let lastScroll = 0;
window.addEventListener('scroll', function () {
    const currentScroll = window.pageYOffset;

    // Scroll progress bar
    const winScroll = document.documentElement.scrollTop;
    const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    const scrolled = (winScroll / height) * 100;
    scrollIndicator.style.width = scrolled + '%';

    // Toggle chatbot button style
    if (currentScroll > 100) {
        chatbotToggle.classList.add('scrolled');
    } else {
        chatbotToggle.classList.remove('scrolled');
    }

    animateOnScroll();
    lastScroll = currentScroll;
});

// Animate elements on scroll
function animateOnScroll() {
    const cards = document.querySelectorAll('.feature-card');
    cards.forEach(card => {
        const cardTop = card.getBoundingClientRect().top;
        const cardBottom = card.getBoundingClientRect().bottom;
        if (cardTop < window.innerHeight - 100 && cardBottom > 0) {
            card.classList.add('visible');
        }
    });
}

animateOnScroll();

// Toast notification
function showToast(message, type = 'info') {
    const existingToast = document.querySelector('.toast-notification');
    if (existingToast) existingToast.remove();

    const toast = document.createElement('div');
    toast.className = `toast-notification toast-${type}`;

    const icons = {
        success: '✅',
        error: '❌',
        warning: '⚠️',
        info: 'ℹ️'
    };

    toast.innerHTML = `
        <span class="toast-icon">${icons[type] || icons.info}</span>
        <span class="toast-message">${message}</span>
    `;

    document.body.appendChild(toast);
    setTimeout(() => toast.classList.add('show'), 10);
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Confirmation modal
function showConfirmModal(message, onConfirm) {
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';

    const modal = document.createElement('div');
    modal.className = 'confirm-modal';
    modal.innerHTML = `
        <div class="modal-icon">❓</div>
        <div class="modal-message">${message}</div>
        <div class="modal-buttons">
            <button class="modal-btn modal-btn-cancel" onclick="closeConfirmModal()">Cancel</button>
            <button class="modal-btn modal-btn-confirm" onclick="confirmAction()">Confirm</button>
        </div>
    `;

    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    setTimeout(() => overlay.classList.add('show'), 10);
    window.confirmCallback = onConfirm;

    overlay.addEventListener('click', e => {
        if (e.target === overlay) closeConfirmModal();
    });
}

window.closeConfirmModal = function () {
    const overlay = document.querySelector('.modal-overlay');
    if (overlay) {
        overlay.classList.remove('show');
        setTimeout(() => overlay.remove(), 300);
    }
    window.confirmCallback = null;
};

window.confirmAction = function () {
    if (window.confirmCallback) window.confirmCallback();
    closeConfirmModal();
};

// Toggle chatbot
function toggleChatbot() {
    chatbotContainer.classList.toggle('active');
    chatbotToggle.classList.toggle('active');
    if (chatbotContainer.classList.contains('active')) {
        setTimeout(() => questionInput.focus(), 300);
    }
}

// Close chatbot when clicking outside
document.addEventListener('click', function (event) {
    const isClickInside =
        chatbotContainer.contains(event.target) ||
        chatbotToggle.contains(event.target);

    if (!isClickInside && chatbotContainer.classList.contains('active')) {
        toggleChatbot();
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

// Initial load
loadHistory();
checkServerStatus();

// Check server status
async function checkServerStatus() {
    try {
        const res = await fetch('/status');
        const data = await res.json();

        if (data.documents_loaded === 0) {
            questionInput.disabled = true;
            askBtn.disabled = true;
            response.textContent = '⚠️ No documents loaded. Please add files to the "files" folder and restart the server.';
        } else {
            questionInput.disabled = false;
            askBtn.disabled = false;
        }
    } catch (error) {
        console.error('Error checking server status:', error);
        showToast('Unable to connect to server', 'error');
    }
}

// Ask question
async function askQuestion() {
    const question = questionInput.value.trim();
    if (!question) {
        showToast('Please enter a question', 'warning');
        return;
    }

    askBtn.disabled = true;
    askBtn.innerHTML = '<span class="loading"></span>';
    response.textContent = '🤔 Thinking...';
    sourcesDiv.style.display = 'none';

    try {
        const res = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });

        const data = await res.json();

        if (res.ok) {
            response.textContent = data.answer;
            if (data.sources?.length) {
                sourcesDiv.innerHTML = `<strong>📚 Sources:</strong> ${data.sources.join(', ')}`;
                sourcesDiv.style.display = 'block';
            }
            questionInput.value = '';
            showToast('Answer received!', 'success');

            if (document.getElementById('historyTab').classList.contains('active')) {
                loadHistory();
            }
        } else {
            response.textContent = data.message || 'Failed to get answer';
            showToast(data.message || 'Failed to get answer', 'error');
        }
    } catch (error) {
        response.textContent = `Error: ${error.message}`;
        showToast(`Error: ${error.message}`, 'error');
    } finally {
        askBtn.disabled = false;
        askBtn.textContent = 'Ask';
    }
}

// Load history
async function loadHistory() {
    try {
        const res = await fetch('/history');
        const data = await res.json();

        if (data.history?.length) {
            historyList.innerHTML = data.history.reverse().map(item => {
                const date = new Date(item.timestamp).toLocaleString();
                const sources = item.sources?.length
                    ? `<div class="history-meta">📚 Sources: ${item.sources.join(', ')}</div>`
                    : '';
                return `
                    <div class="history-item">
                        <div class="history-question">Q: ${item.question}</div>
                        <div class="history-answer">A: ${item.answer}</div>
                        ${sources}
                        <div class="history-meta">🕒 ${date}</div>
                    </div>
                `;
            }).join('');
        } else {
            historyList.innerHTML = '<div class="empty-state">No conversation history yet</div>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
        showToast('Failed to load history', 'error');
    }
}

// Clear history
async function clearHistory() {
    showConfirmModal('Are you sure you want to clear all conversation history?', async () => {
        try {
            const res = await fetch('/history', { method: 'DELETE' });
            if (res.ok) {
                loadHistory();
                showToast('History cleared successfully', 'success');
            } else {
                showToast('Failed to clear history', 'error');
            }
        } catch (error) {
            showToast(`Error: ${error.message}`, 'error');
        }
    });
}
