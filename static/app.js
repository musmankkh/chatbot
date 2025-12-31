const questionInput = document.getElementById('questionInput');
const askBtn = document.getElementById('askBtn');
const response = document.getElementById('response');
const historyList = document.getElementById('historyList');
const chatbotToggle = document.getElementById('chatbotToggle');
const chatbotContainer = document.getElementById('chatbotContainer');
const scrollIndicator = document.getElementById('scrollIndicator');

// ---------------- SCROLL EFFECT ----------------
window.addEventListener('scroll', () => {
  const winScroll = document.documentElement.scrollTop;
  const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
  scrollIndicator.style.width = (winScroll / height) * 100 + '%';
});

// ---------------- TOAST ----------------
function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast-notification toast-${type}`;
  toast.innerHTML = `<span>${message}</span>`;
  document.body.appendChild(toast);

  setTimeout(() => toast.classList.add('show'), 50);
  setTimeout(() => toast.remove(), 3000);
}

// ---------------- TOGGLE CHAT ----------------
function toggleChatbot() {
  chatbotContainer.classList.toggle('active');
  if (chatbotContainer.classList.contains('active')) {
    setTimeout(() => questionInput.focus(), 300);
  }
}

// ---------------- ASK QUESTION ----------------
async function askQuestion() {
  const question = questionInput.value.trim();
  if (!question) {
    showToast('Please enter a question', 'warning');
    return;
  }

  askBtn.disabled = true;
  response.textContent = '🤔 Thinking...';

  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    });

    const data = await res.json();

    if (!res.ok) {
      response.textContent = data.error || 'Something went wrong';
      showToast('Request failed', 'error');
      return;
    }

    response.textContent = data.answer;
    questionInput.value = '';
    showToast('Answer received', 'success');

    loadHistory();
  } catch (err) {
    response.textContent = 'Error contacting server';
    showToast('Server error', 'error');
  } finally {
    askBtn.disabled = false;
  }
}

// ---------------- HISTORY ----------------
async function loadHistory() {
  try {
    const res = await fetch('/history');
    const data = await res.json();

    if (!data.history || data.history.length === 0) {
      historyList.innerHTML = '<p class="empty-state">No history yet</p>';
      return;
    }

    historyList.innerHTML = data.history.reverse().map(item => `
      <div class="history-item">
        <div class="history-question">Q: ${item.question}</div>
        <div class="history-answer">A: ${item.answer}</div>
        <div class="history-meta">${new Date(item.timestamp).toLocaleString()}</div>
      </div>
    `).join('');
  } catch (e) {
    showToast('Failed to load history', 'error');
  }
}

// ---------------- INITIAL LOAD ----------------
document.addEventListener('DOMContentLoaded', () => {
  loadHistory();
});
