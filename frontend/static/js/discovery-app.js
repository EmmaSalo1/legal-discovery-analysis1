// Legal Discovery Analysis Dashboard - Complete JavaScript Application
class LegalDiscoveryApp {
    constructor() {
        this.caseId = this.extractCaseId();
        this.websocket = null;
        this.currentFiles = [];
        this.processingQueue = new Map();
        
        this.initializeEventListeners();
        this.connectWebSocket();
    }
    
    extractCaseId() {
        const path = window.location.pathname;
        const matches = path.match(/\/case\/([^\/]+)/);
        return matches ? matches[1] : null;
    }
    
    initializeEventListeners() {
        // File upload drag and drop
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');
        
        if (uploadArea && fileInput) {
            uploadArea.addEventListener('click', () => fileInput.click());
            uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
            uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
            uploadArea.addEventListener('drop', this.handleDrop.bind(this));
            fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        }
        
        // Chat functionality
        const chatInput = document.getElementById('chat-input');
        const chatSend = document.getElementById('chat-send');
        
        if (chatInput && chatSend) {
            chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') this.sendMessage();
            });
            chatSend.addEventListener('click', this.sendMessage.bind(this));
        }
        
        // Upload form
        const uploadForm = document.getElementById('upload-form');
        if (uploadForm) {
            uploadForm.addEventListener('submit', this.handleFormUpload.bind(this));
        }
        
        // Modal close on outside click
        window.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal')) {
                e.target.style.display = 'none';
            }
        });
    }
    
    connectWebSocket() {
        if (!this.caseId) return;
        
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/chat/${this.caseId}`;
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = () => {
            console.log('WebSocket connected');
            this.updateChatStatus('Connected');
        };
        
        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
        
        this.websocket.onclose = () => {
            console.log('WebSocket disconnected');
            this.updateChatStatus('Disconnected');
            // Attempt to reconnect after 5 seconds
            setTimeout(() => this.connectWebSocket(), 5000);
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateChatStatus('Error');
        };
    }
    
    handleDragOver(e) {
        e.preventDefault();
        e.currentTarget.classList.add('dragover');
    }
    
    handleDragLeave(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
    }
    
    handleDrop(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
        
        const files = Array.from(e.dataTransfer.files);
        this.uploadFiles(files);
    }
    
    handleFileSelect(e) {
        const files = Array.from(e.target.files);
        this.uploadFiles(files);
        e.target.value = ''; // Reset input
    }
    
    handleFormUpload(e) {
        e.preventDefault();
        
        const formData = new FormData(e.target);
        const files = Array.from(formData.getAll('files'));
        
        if (files.length === 0) {
            this.showNotification('Please select files to upload', 'warning');
            return;
        }
        
        this.uploadFiles(files, {
            party: formData.get('party'),
            document_type: formData.get('document_type')
        });
        
        this.closeUploadModal();
        e.target.reset();
    }
    
    async uploadFiles(files, options = {}) {
        if (!this.caseId) {
            this.showNotification('Please select a case first', 'warning');
            return;
        }
        
        const formData = new FormData();
        
        files.forEach(file => {
            formData.append('files', file);
        });
        
        formData.append('party', options.party || 'plaintiff_production');
        formData.append('document_type', options.document_type || 'general');
        
        try {
            this.showNotification(`Uploading ${files.length} file(s)...`, 'info');
            
            const response = await fetch(`/api/cases/${this.caseId}/documents/upload`, {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.showNotification(`Uploaded ${files.length} file(s) successfully`, 'success');
                
                // Add to processing queue display
                result.files.forEach(file => {
                    this.addToProcessingQueue(file);
                });
                
                this.refreshFileList();
            } else {
                throw new Error(result.detail || 'Upload failed');
            }
        } catch (error) {
            console.error('Upload error:', error);
            this.showNotification(`Upload failed: ${error.message}`, 'error');
        }
    }
    
    addToProcessingQueue(file) {
        this.processingQueue.set(file.processing_id, {
            ...file,
            status: 'queued',
            progress: 0
        });
        this.updateProcessingDisplay();
    }
    
    updateProcessingDisplay() {
        const queueContainer = document.getElementById('processing-queue');
        if (!queueContainer) return;
        
        if (this.processingQueue.size === 0) {
            queueContainer.innerHTML = '<p class="no-processing">No files currently processing</p>';
            return;
        }
        
        const queueHtml = Array.from(this.processingQueue.values()).map(item => `
            <div class="processing-item" data-id="${item.processing_id}">
                <div class="processing-info">
                    <div class="processing-filename">${item.filename}</div>
                    <div class="processing-status">${item.status}</div>
                </div>
                <div class="processing-type">
                    <span class="file-type ${item.file_type}">${item.file_type}</span>
                </div>
            </div>
        `).join('');
        
        queueContainer.innerHTML = queueHtml;
    }
    
    handleWebSocketMessage(data) {
    if (data.type === 'response') {
        this.addMessageToChat(data.message, 'bot', data.sources, data.file_sources);
    } else if (data.type === 'error') {
        this.addMessageToChat(data.message, 'bot', [], [], 'error');
    } else if (data.type === 'processing_complete') {
        this.handleProcessingComplete(data);
    }
    }
    
    handleProcessingComplete(data) {
        const item = this.processingQueue.get(data.processing_id);
        if (item) {
            if (data.status === 'completed') {
                this.processingQueue.delete(data.processing_id);
                this.showNotification(`Processing completed: ${item.filename}`, 'success');
            } else {
                item.status = 'failed';
                this.showNotification(`Processing failed: ${item.filename}`, 'error');
            }
            this.updateProcessingDisplay();
            this.refreshFileList();
        }
    }
    
    async sendMessage() {
        const chatInput = document.getElementById('chat-input');
        const message = chatInput.value.trim();
        
        if (!message || !this.websocket) return;
        
        // Add user message to chat
        this.addMessageToChat(message, 'user');
        chatInput.value = '';
        
        // Send to WebSocket
        this.websocket.send(JSON.stringify({
            message: message,
            type: 'general'
        }));
    }
    
addMessageToChat(message, sender, sources = [], fileSources = [], type = 'normal') {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    let messageHtml = `<p>${message}</p>`;
    
    // Add enhanced file sources with links and previews
    if (fileSources && fileSources.length > 0) {
        messageHtml += '<div class="file-sources"><strong>📁 Referenced Files:</strong>';
        messageHtml += '<div class="file-source-list">';
        
        fileSources.forEach(file => {
            const fileIcon = this.getFileIcon(file.type);
            const fileName = file.name || 'Unknown file';
            const contentType = file.content_type || 'content';
            const confidence = file.confidence ? ` (${(file.confidence * 100).toFixed(0)}% confidence)` : '';
            
            messageHtml += `
                <div class="file-source-item">
                    <div class="file-source-header">
                        <span class="file-icon">${fileIcon}</span>
                        <button class="file-link" onclick="app.viewFileAnalysis('${file.id}')" title="View analysis">
                            ${fileName}
                        </button>
                        <span class="file-type-badge ${file.type}">${file.type}</span>
                    </div>
                    <div class="file-source-meta">
                        ${contentType}${confidence}
                    </div>
                </div>
            `;
        });
        
        messageHtml += '</div></div>';
    }
    
    // Add basic sources fallback (if no enhanced file sources)
    else if (sources && sources.length > 0) {
        messageHtml += '<div class="message-sources"><strong>📄 Sources:</strong><ul>';
        sources.forEach(source => {
            const filename = source.split('/').pop();
            messageHtml += `<li>${filename}</li>`;
        });
        messageHtml += '</ul></div>';
    }
    
    messageDiv.innerHTML = messageHtml;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Add this new method to your LegalDiscoveryApp class
getFileIcon(fileType) {
    const icons = {
        'audio': '🎵',
        'video': '🎥',
        'image': '📷',
        'document': '📄'
    };
    return icons[fileType] || '📄';
}

// Add this new method to your LegalDiscoveryApp class
async viewFileAnalysis(fileId) {
    if (!this.caseId || !fileId) {
        this.showNotification('Unable to view file analysis', 'error');
        return;
    }
    
    try {
        const response = await fetch(`/api/cases/${this.caseId}/files/${fileId}/analysis`);
        
        if (response.ok) {
            const analysis = await response.json();
            this.showFileAnalysisModal(analysis);
        } else {
            this.showNotification('Analysis not available for this file', 'warning');
        }
    } catch (error) {
        console.error('Error getting file analysis:', error);
        this.showNotification('Error loading file analysis', 'error');
    }
}

// Add this new method to your LegalDiscoveryApp class
showFileAnalysisModal(analysis) {
    // Create analysis modal
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h2>📄 File Analysis</h2>
                <span class="close" onclick="this.closest('.modal').remove()">&times;</span>
            </div>
            <div class="modal-body">
                <div class="analysis-content">
                    ${this.formatFileAnalysis(analysis)}
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    modal.style.display = 'block';
    
    // Close on outside click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.remove();
        }
    });
}

// Add this new method to your LegalDiscoveryApp class
formatFileAnalysis(analysis) {
    let html = `
        <div class="analysis-header">
            <h3>${analysis.metadata?.filename || 'File Analysis'}</h3>
            <span class="file-type ${analysis.file_type}">${analysis.file_type}</span>
        </div>
    `;
    
    // File metadata
    if (analysis.metadata) {
        html += `
            <div class="analysis-section">
                <h4>📊 File Information</h4>
                <div class="metadata-grid">
                    <div><strong>Size:</strong> ${this.formatFileSize(analysis.metadata.file_size || 0)}</div>
                    <div><strong>Format:</strong> ${analysis.metadata.format || 'Unknown'}</div>
                    ${analysis.metadata.duration ? `<div><strong>Duration:</strong> ${Math.round(analysis.metadata.duration)}s</div>` : ''}
                    ${analysis.metadata.resolution ? `<div><strong>Resolution:</strong> ${analysis.metadata.resolution}</div>` : ''}
                </div>
            </div>
        `;
    }
    
    // Content based on file type
    if (analysis.file_type === 'audio' && analysis.transcript) {
        html += `
            <div class="analysis-section">
                <h4>🎵 Audio Transcript</h4>
                <div class="confidence-score">Confidence: ${(analysis.transcript.confidence * 100).toFixed(1)}%</div>
                <div class="transcript-text">${analysis.transcript.text}</div>
            </div>
        `;
    }
    
    if (analysis.file_type === 'video' && analysis.audio_analysis?.transcript) {
        html += `
            <div class="analysis-section">
                <h4>🎥 Video Audio Transcript</h4>
                <div class="confidence-score">Confidence: ${(analysis.audio_analysis.transcript.confidence * 100).toFixed(1)}%</div>
                <div class="transcript-text">${analysis.audio_analysis.transcript.text}</div>
            </div>
        `;
    }
    
    if (analysis.file_type === 'image' && analysis.ocr_results) {
        html += `
            <div class="analysis-section">
                <h4>📷 OCR Text</h4>
                <div class="confidence-score">Confidence: ${analysis.ocr_results.total_confidence.toFixed(1)}%</div>
                <div class="ocr-text">${analysis.ocr_results.combined_text}</div>
            </div>
        `;
    }
    
    if (analysis.file_type === 'document' && analysis.content) {
        html += `
            <div class="analysis-section">
                <h4>📄 Document Content</h4>
                <div class="document-content">${analysis.content.substring(0, 1000)}${analysis.content.length > 1000 ? '...' : ''}</div>
            </div>
        `;
    }
    
    // Entities
    if (analysis.entities) {
        html += `
            <div class="analysis-section">
                <h4>🔍 Extracted Information</h4>
                <div class="entities-grid">
        `;
        
        Object.entries(analysis.entities).forEach(([type, items]) => {
            if (items.length > 0) {
                html += `
                    <div class="entity-group">
                        <strong>${type.replace('_', ' ').toUpperCase()}:</strong>
                        <span class="entity-list">${items.slice(0, 5).join(', ')}${items.length > 5 ? '...' : ''}</span>
                    </div>
                `;
            }
        });
        
        html += '</div></div>';
    }
    
    // Privilege flags
    if (analysis.privilege_flags && analysis.privilege_flags.length > 0) {
        html += `
            <div class="analysis-section privilege-warning">
                <h4>⚠️ Privilege Alerts</h4>
                ${analysis.privilege_flags.map(flag => `
                    <div class="privilege-flag">
                        <strong>${flag.type}:</strong> "${flag.match}"
                        <div class="privilege-context">${flag.context}</div>
                    </div>
                `).join('')}
            </div>
        `;
    }
    
    // Summary
    if (analysis.summary) {
        html += `
            <div class="analysis-section">
                <h4>📋 Summary</h4>
                <p>${analysis.summary}</p>
            </div>
        `;
    }
    
    return html || '<p>No detailed analysis available.</p>';
}
    
    updateChatStatus(status) {
        const statusElement = document.getElementById('chat-status');
        if (statusElement) {
            statusElement.textContent = status;
            statusElement.className = 'chat-status ' + (status === 'Connected' ? 'connected' : 'disconnected');
        }
    }
    
    async refreshFileList() {
        if (!this.caseId) return;
        
        try {
            const response = await fetch(`/api/cases/${this.caseId}/files`);
            const data = await response.json();
            
            this.currentFiles = data.files;
            this.updateFileDisplay();
            this.updateStats();
        } catch (error) {
            console.error('Error refreshing file list:', error);
            this.showNotification('Error loading files', 'error');
        }
    }
    
    updateFileDisplay() {
        const filesGrid = document.getElementById('files-grid');
        if (!filesGrid) return;
        
        const filterType = document.getElementById('file-type-filter')?.value || '';
        const filteredFiles = filterType ? 
            this.currentFiles.filter(file => file.type === filterType) : 
            this.currentFiles;
        
        if (filteredFiles.length === 0) {
            filesGrid.innerHTML = '<div class="no-files"><p>No files found</p></div>';
            return;
        }
        
        const filesHtml = filteredFiles.map(file => this.createFileCard(file)).join('');
        filesGrid.innerHTML = filesHtml;
    }
    
    createFileCard(file) {
        const sizeFormatted = this.formatFileSize(file.size);
        const dateFormatted = new Date(file.modified).toLocaleString();
        
        return `
            <div class="file-card" data-file-type="${file.type}">
                <div class="file-header">
                    <div class="file-name" title="${file.filename}">${file.filename}</div>
                    <span class="file-type ${file.type}">${file.type}</span>
                </div>
                <div class="file-meta">
                    <div>Size: ${sizeFormatted}</div>
                    <div>Modified: ${dateFormatted}</div>
                    <div>Path: ${file.path}</div>
                </div>
                <div class="file-actions">
                    <button onclick="app.analyzeFile('${file.path.replace(/'/g, "\\'")}')" class="btn-secondary btn-small">Analyze</button>
                    <button onclick="app.chatAboutFile('${file.filename.replace(/'/g, "\\'")}')" class="btn-secondary btn-small">Ask AI</button>
                </div>
            </div>
        `;
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    updateStats() {
        const stats = this.calculateStats();
        
        document.getElementById('total-files').textContent = stats.total;
        document.getElementById('audio-files').textContent = stats.audio;
        document.getElementById('video-files').textContent = stats.video;
        document.getElementById('image-files').textContent = stats.image;
        document.getElementById('document-files').textContent = stats.document;
        
        const processingStatus = this.processingQueue.size > 0 ? 
            `Processing ${this.processingQueue.size}` : 'Ready';
        document.getElementById('processing-status').textContent = processingStatus;
    }
    
    calculateStats() {
        const stats = {
            total: this.currentFiles.length,
            audio: 0,
            video: 0,
            image: 0,
            document: 0
        };
        
        this.currentFiles.forEach(file => {
            if (stats.hasOwnProperty(file.type)) {
                stats[file.type]++;
            }
        });
        
        return stats;
    }
    
    async analyzeFile(filePath) {
        this.showNotification('Detailed analysis feature coming soon', 'info');
    }
    
    chatAboutFile(filename) {
        const chatInput = document.getElementById('chat-input');
        chatInput.value = `Tell me about the file "${filename}"`;
        chatInput.focus();
    }
    
    async performSearch() {
        const query = document.getElementById('search-input').value.trim();
        if (!query || !this.caseId) {
            this.showNotification('Please enter a search query', 'warning');
            return;
        }
        
        const checkboxes = document.querySelectorAll('#search-modal input[type="checkbox"]:checked');
        const fileTypes = Array.from(checkboxes).map(cb => cb.value).join(',');
        
        try {
            this.showNotification('Searching...', 'info');
            
            const response = await fetch(`/api/cases/${this.caseId}/search?query=${encodeURIComponent(query)}&file_types=${fileTypes}`);
            const results = await response.json();
            
            this.displaySearchResults(results);
            this.showNotification(`Found ${results.total_results} results`, 'success');
        } catch (error) {
            console.error('Search error:', error);
            this.showNotification('Search failed', 'error');
        }
    }
    
    displaySearchResults(results) {
        const resultsContainer = document.getElementById('search-results');
        if (!resultsContainer) return;
        
        if (results.results.length === 0) {
            resultsContainer.innerHTML = '<div class="no-results"><p>No results found</p></div>';
            return;
        }
        
        const resultsHtml = results.results.map(result => `
            <div class="search-result">
                <div class="result-header">
                    <span class="result-filename">${result.file_path.split('/').pop()}</span>
                    <span class="result-type ${result.file_type}">${result.file_type}</span>
                </div>
                <div class="result-snippets">
                    ${result.snippets.map(snippet => `
                        <div class="snippet">
                            <strong>${snippet.type}:</strong> ${snippet.text}
                        </div>
                    `).join('')}
                </div>
                <div class="result-score">
                    Relevance: ${(result.relevance_score * 100).toFixed(1)}%
                </div>
            </div>
        `).join('');
        
        resultsContainer.innerHTML = `
            <div class="results-header">
                <h3>Search Results (${results.total_results})</h3>
            </div>
            <div class="results-list">
                ${resultsHtml}
            </div>
        `;
    }
    
    showNotification(message, type = 'info') {
        // Remove existing notifications
        const existing = document.querySelectorAll('.notification');
        existing.forEach(n => n.remove());
        
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 1rem 1.5rem;
            border-radius: 4px;
            color: white;
            z-index: 1001;
            max-width: 300px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            font-size: 0.9rem;
            animation: slideIn 0.3s ease;
        `;
        
        // Set background color based on type
        const colors = {
            success: '#27ae60',
            error: '#e74c3c',
            warning: '#f39c12',
            info: '#3498db'
        };
        notification.style.backgroundColor = colors[type] || colors.info;
        
        notification.textContent = message;
        document.body.appendChild(notification);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.style.animation = 'slideOut 0.3s ease';
                setTimeout(() => notification.remove(), 300);
            }
        }, 5000);
    }
    
    filterFiles() {
        this.updateFileDisplay();
    }
    
    closeUploadModal() {
        const modal = document.getElementById('upload-modal');
        if (modal) modal.style.display = 'none';
    }
    
    closeSearchModal() {
        const modal = document.getElementById('search-modal');
        if (modal) modal.style.display = 'none';
    }
}

// Global functions for HTML onclick handlers
function showUploadModal() {
    const modal = document.getElementById('upload-modal');
    if (modal) modal.style.display = 'block';
}

function closeUploadModal() {
    if (window.app) window.app.closeUploadModal();
}

function showSearchModal() {
    const modal = document.getElementById('search-modal');
    if (modal) modal.style.display = 'block';
}

function closeSearchModal() {
    if (window.app) window.app.closeSearchModal();
}

function performSearch() {
    if (window.app) window.app.performSearch();
}

function filterFiles() {
    if (window.app) window.app.filterFiles();
}

function refreshFileList() {
    if (window.app) window.app.refreshFileList();
}

function sendMessage() {
    if (window.app) window.app.sendMessage();
}

function initializeDashboard() {
    window.app = new LegalDiscoveryApp();
    
    // Load initial data
    if (window.app.caseId) {
        window.app.refreshFileList();
    } else {
        // Prompt to create case if none exists
        const caseNumber = prompt('Enter case number to create new case:');
        if (caseNumber) {
            createCase(caseNumber);
        }
    }
}

async function createCase(caseNumber) {
    try {
        const response = await fetch('/api/cases/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                case_number: caseNumber,
                case_name: `Case ${caseNumber}`,
                description: 'Legal discovery analysis case'
            })
        });
        
        if (response.ok) {
            window.location.href = `/case/${caseNumber}`;
        } else {
            const error = await response.json();
            alert(`Error creating case: ${error.detail}`);
        }
    } catch (error) {
        console.error('Error creating case:', error);
        alert('Error creating case');
    }
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    
    .no-files, .no-results {
        text-align: center;
        padding: 2rem;
        color: #7f8c8d;
        font-style: italic;
    }
    
    .search-result {
        border: 1px solid #ecf0f1;
        border-radius: 4px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    
    .result-filename {
        font-weight: 600;
        color: var(--primary-color);
    }
    
    .snippet {
        background: #f8f9fa;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    
    .result-score {
        text-align: right;
        font-size: 0.8rem;
        color: #7f8c8d;
        margin-top: 0.5rem;
    }
    
    .connected {
        color: #27ae60;
    }
    
    .disconnected {
        color: #e74c3c;
    }
    
    .message-sources {
        margin-top: 0.5rem;
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    .message-sources ul {
        margin: 0.25rem 0 0 1rem;
    }
    
    .btn-small {
        padding: 0.25rem 0.5rem;
        font-size: 0.8rem;
        margin-right: 0.25rem;
    }
    
    .file-actions {
        margin-top: 0.5rem;
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
`;
document.head.appendChild(style);

