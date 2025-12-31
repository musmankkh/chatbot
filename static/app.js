// Elements
const questionInput = document.getElementById('questionInput');
const askBtn = document.getElementById('askBtn');
const response = document.getElementById('response');
const sourcesDiv = document.getElementById('sources');
const historyList = document.getElementById('historyList');
const chatbotToggle = document.getElementById('chatbotToggle');
const chatbotContainer = document.getElementById('chatbotContainer');
const scrollIndicator = document.getElementById('scrollIndicator');

// Scroll Event Listener
let lastScroll = 0;
window.addEventListener('scroll', function() {
  const currentScroll = window.pageYOffset;
  
  // Update scroll progress indicator
  const winScroll = document.documentElement.scrollTop;
  const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
  const scrolled = (winScroll / height) * 100;
  scrollIndicator.style.width = scrolled + '%';
  
  // Add/remove scrolled class to chatbot button
  if (currentScroll > 100) {
    chatbotToggle.classList.add('scrolled');
  } else {
    chatbotToggle.classList.remove('scrolled');
  }
  
  // Animate feature cards on scroll
  animateOnScroll();
  
  lastScroll = currentScroll;
});

// Animate elements when they come into view
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

// Initial animation check
animateOnScroll();

// Toast notification
function showToast(message, type = 'info') {
  const existingToast = document.querySelector('.toast-notification');
  if (existingToast) existingToast.remove();

  const toast = document.createElement('div');
  toast.className = `toast-notification toast-${type}`;
  
  const icons = { success: '✅', error: '❌', warning: '⚠️', info: 'ℹ️' };
  
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
  
  overlay.addEventListener('click', function(e) {
    if (e.target === overlay) closeConfirmModal();
  });
}

window.closeConfirmModal = function() {
  const overlay = document.querySelector('.modal-overlay');
  if (overlay) {
    overlay.classList.remove('show');
    setTimeout(() => overlay.remove(), 300);
  }
  window.confirmCallback = null;
}

window.confirmAction = function() {
  if (window.confirmCallback) window.confirmCallback();
  closeConfirmModal();
}

// Toggle chatbot
function toggleChatbot() {
  chatbotContainer.classList.toggle('active');
  chatbotToggle.classList.toggle('active');
  
  if (chatbotContainer.classList.contains('active')) {
    setTimeout(() => questionInput.focus(), 300);
  }
}

// Close chatbot when clicking outside
document.addEventListener('click', function(event) {
  const isClickInside = chatbotContainer.contains(event.target) || 
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

// Mock functions for demo
function askQuestion() {
  const question = questionInput.value.trim();
  if (!question) {
    showToast('Please enter a question', 'warning');
    return;
  }

  askBtn.disabled = true;
  askBtn.innerHTML = '<span class="loading"></span>';
  response.textContent = '🤔 Thinking...';
  sourcesDiv.style.display = 'none';

  // Simulate API call
  setTimeout(() => {
    response.textContent = `This is a demo answer to: "${question}". In production, this would connect to your backend API.`;
    sourcesDiv.innerHTML = '<strong>📚 Sources:</strong> document1.pdf, document2.txt';
    sourcesDiv.style.display = 'block';
    questionInput.value = '';
    askBtn.disabled = false;
    askBtn.textContent = 'Ask';
    showToast('Answer received!', 'success');
  }, 1500);
}

function loadHistory() {
  historyList.innerHTML = `
    <div class="history-item">
      <div class="history-question">Q: What is machine learning?</div>
      <div class="history-answer">A: Machine learning is a subset of artificial intelligence...</div>
      <div class="history-meta">📚 Sources: ml_guide.pdf</div>
      <div class="history-meta">🕒 ${new Date().toLocaleString()}</div>
    </div>
  `;
}

function clearHistory() {
  showConfirmModal('Are you sure you want to clear all conversation history?', function() {
    historyList.innerHTML = '<div class="empty-state">No conversation history yet</div>';
    showToast('History cleared successfully', 'success');
  });
}