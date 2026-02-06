// Configuration - DataLoader service endpoint
const API_BASE_URL = 'http://192.168.86.28:30800';

let statusCheckInterval = null;
let isLoading = false;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    loadIndices();
    
    // Check health every 30 seconds
    setInterval(checkHealth, 30000);
});

// Toast notifications
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 5000);
}

// API helper function
async function apiCall(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'API request failed');
        }
        
        return await response.json();
    } catch (error) {
        if (error.message.includes('Failed to fetch')) {
            throw new Error('Unable to connect to Data Loader service');
        }
        throw error;
    }
}

// Check system health
async function checkHealth() {
    const esStatus = document.querySelector('#es-status .status-badge');
    const loaderStatus = document.querySelector('#loader-status .status-badge');
    
    try {
        const health = await apiCall('/health');
        
        // Update loader status
        loaderStatus.textContent = 'Healthy';
        loaderStatus.className = 'status-badge healthy';
        
        // Update Elasticsearch status
        if (health.elasticsearch && health.elasticsearch.status !== 'unavailable') {
            const status = health.elasticsearch.status;
            esStatus.textContent = status.charAt(0).toUpperCase() + status.slice(1);
            esStatus.className = `status-badge ${status}`;
        } else {
            esStatus.textContent = 'Unavailable';
            esStatus.className = 'status-badge unhealthy';
        }
    } catch (error) {
        loaderStatus.textContent = 'Unavailable';
        loaderStatus.className = 'status-badge unhealthy';
        esStatus.textContent = 'Unknown';
        esStatus.className = 'status-badge checking';
    }
}

// Load and display indices
async function loadIndices() {
    const container = document.getElementById('indices-list');
    const deleteSelect = document.getElementById('delete-index');
    const searchSelect = document.getElementById('search-index');
    
    container.innerHTML = '<p class="loading">Loading indices...</p>';
    
    try {
        const result = await apiCall('/indices');
        
        // Clear and update delete dropdown
        deleteSelect.innerHTML = '<option value="__all__">All Indices</option>';
        searchSelect.innerHTML = '';
        
        if (result.indices && result.indices.length > 0) {
            container.innerHTML = result.indices.map(idx => `
                <div class="index-card">
                    <h4>📁 ${idx.name}</h4>
                    <div class="index-stats">
                        <span>📄 ${formatNumber(idx.docs_count)} docs</span>
                        <span>💾 ${idx.store_size}</span>
                        <span class="status-badge ${idx.health}">${idx.health}</span>
                    </div>
                </div>
            `).join('');
            
            result.indices.forEach(idx => {
                deleteSelect.innerHTML += `<option value="${idx.name}">${idx.name}</option>`;
                searchSelect.innerHTML += `<option value="${idx.name}">${idx.name}</option>`;
            });
        } else {
            container.innerHTML = '<div class="no-indices">No indices found. Load some data to get started!</div>';
            searchSelect.innerHTML = '<option value="msmarco">msmarco</option>';
        }
    } catch (error) {
        container.innerHTML = `<div class="no-indices">Error loading indices: ${error.message}</div>`;
        showToast(error.message, 'error');
    }
}

// Pre-load the embedding model
async function preloadModel() {
    const preloadBtn = document.getElementById('preload-btn');
    
    try {
        preloadBtn.disabled = true;
        preloadBtn.innerHTML = '⏳ Loading Model...';
        
        const result = await apiCall('/model/load', { method: 'POST' });
        
        showToast(`Model ${result.model_name} loaded successfully!`, 'success');
    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        preloadBtn.disabled = false;
        preloadBtn.innerHTML = '🧠 Pre-load Model';
    }
}

// Load MS MARCO data
async function loadData() {
    if (isLoading) {
        showToast('A loading operation is already in progress', 'warning');
        return;
    }
    
    const indexName = document.getElementById('index-name').value || 'msmarco';
    const maxDocs = parseInt(document.getElementById('max-docs').value) || null;
    const batchSize = parseInt(document.getElementById('batch-size').value) || 500;
    const chunkSize = parseInt(document.getElementById('chunk-size').value) || 512;
    const chunkOverlap = parseInt(document.getElementById('chunk-overlap').value) || 50;
    const embeddingBatch = parseInt(document.getElementById('embedding-batch').value) || 32;
    
    const loadBtn = document.getElementById('load-btn');
    const progressContainer = document.getElementById('progress-container');
    
    try {
        loadBtn.disabled = true;
        loadBtn.innerHTML = '⏳ Starting...';
        progressContainer.style.display = 'block';
        
        const config = {
            index_name: indexName,
            dataset_split: 'train',
            batch_size: batchSize,
            chunk_size: chunkSize,
            chunk_overlap: chunkOverlap,
            embedding_batch_size: embeddingBatch
        };
        
        if (maxDocs && maxDocs > 0) {
            config.max_documents = maxDocs;
        }
        
        await apiCall('/load/msmarco', {
            method: 'POST',
            body: JSON.stringify(config)
        });
        
        isLoading = true;
        showToast('Data loading with embeddings started!', 'success');
        
        // Start polling for status
        statusCheckInterval = setInterval(checkLoadingStatus, 2000);
        
    } catch (error) {
        loadBtn.disabled = false;
        loadBtn.innerHTML = '📥 Start Loading';
        progressContainer.style.display = 'none';
        showToast(error.message, 'error');
    }
}

// Check loading status
async function checkLoadingStatus() {
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    const loadBtn = document.getElementById('load-btn');
    const progressContainer = document.getElementById('progress-container');
    
    try {
        const status = await apiCall('/status');
        
        progressText.textContent = status.message;
        
        if (status.is_loading) {
            loadBtn.innerHTML = '⏳ Loading...';
            
            // Animate progress (indeterminate if we don't know total)
            if (status.progress > 0) {
                // Use a pseudo-progress based on documents processed
                const progress = Math.min((status.progress / 10000) * 100, 95);
                progressFill.style.width = `${progress}%`;
            } else {
                // Pulse animation for indeterminate state
                progressFill.style.width = '30%';
            }
        } else {
            // Loading complete
            isLoading = false;
            clearInterval(statusCheckInterval);
            
            progressFill.style.width = '100%';
            loadBtn.disabled = false;
            loadBtn.innerHTML = '📥 Start Loading';
            
            if (status.message.includes('Error')) {
                showToast(status.message, 'error');
            } else {
                showToast('Data loading completed!', 'success');
            }
            
            // Refresh indices list
            setTimeout(() => {
                loadIndices();
                progressContainer.style.display = 'none';
            }, 2000);
        }
    } catch (error) {
        console.error('Error checking status:', error);
    }
}

// Clear data
async function clearData() {
    const indexName = document.getElementById('delete-index').value;
    
    const confirmMsg = indexName === '__all__' 
        ? 'Are you sure you want to DELETE ALL indices? This cannot be undone!'
        : `Are you sure you want to delete the "${indexName}" index? This cannot be undone!`;
    
    if (!confirm(confirmMsg)) {
        return;
    }
    
    try {
        const endpoint = indexName === '__all__' ? '/clear' : `/clear/${indexName}`;
        const result = await apiCall(endpoint, { method: 'DELETE' });
        
        showToast(result.message, 'success');
        loadIndices();
    } catch (error) {
        showToast(error.message, 'error');
    }
}

// Search data
async function searchData() {
    const query = document.getElementById('search-query').value.trim();
    const index = document.getElementById('search-index').value;
    const searchType = document.getElementById('search-type').value;
    const resultsContainer = document.getElementById('search-results');
    
    if (!query) {
        showToast('Please enter a search query', 'warning');
        return;
    }
    
    resultsContainer.innerHTML = '<p class="loading">Searching...</p>';
    
    try {
        let result;
        
        if (searchType === 'vector') {
            // Vector (semantic) search
            result = await apiCall('/search/vector', {
                method: 'POST',
                body: JSON.stringify({
                    query: query,
                    index_name: index,
                    size: 10
                })
            });
        } else if (searchType === 'hybrid') {
            // Hybrid search
            result = await apiCall('/search/hybrid', {
                method: 'POST',
                body: JSON.stringify({
                    query: query,
                    index_name: index,
                    size: 10,
                    vector_boost: 0.7,
                    text_boost: 0.3
                })
            });
        } else {
            // Text (BM25) search
            result = await apiCall(`/search/${index}?q=${encodeURIComponent(query)}&size=10`);
        }
        
        if (result.hits && result.hits.length > 0) {
            const searchTypeLabel = {
                'vector': '🧠 Vector (Semantic)',
                'hybrid': '🔀 Hybrid',
                'text': '📝 Text (BM25)'
            };
            
            resultsContainer.innerHTML = `
                <p style="margin-bottom: 15px; color: var(--text-secondary);">
                    Found ${formatNumber(result.total)} results using ${searchTypeLabel[searchType] || result.search_type}
                    ${result.model ? ` • Model: ${result.model}` : ''}
                </p>
                ${result.hits.map(hit => `
                    <div class="search-result-item">
                        <span class="score">Score: ${hit.score.toFixed(4)}</span>
                        <p class="passage">${escapeHtml(hit.source.text || hit.source.passage_text)}</p>
                        ${hit.source.query_text ? `<p class="query">Query: ${escapeHtml(hit.source.query_text)}</p>` : ''}
                        ${hit.source.chunk_index !== undefined ? `<span class="chunk-info">Chunk ${hit.source.chunk_index}</span>` : ''}
                    </div>
                `).join('')}
            `;
        } else {
            resultsContainer.innerHTML = '<div class="no-indices">No results found</div>';
        }
    } catch (error) {
        resultsContainer.innerHTML = `<div class="no-indices">Error: ${error.message}</div>`;
        showToast(error.message, 'error');
    }
}

// Utility functions
function formatNumber(num) {
    if (num === null || num === undefined) return '0';
    return parseInt(num).toLocaleString();
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Allow Enter key for search
document.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && e.target.id === 'search-query') {
        searchData();
    }
});
