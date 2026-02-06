// Configuration - DataLoader service endpoint (stored in localStorage)
const DEFAULT_API_URL = 'http://192.168.86.151:8000';

function getApiBaseUrl() {
    return localStorage.getItem('apiBaseUrl') || DEFAULT_API_URL;
}

function setApiBaseUrl(url) {
    localStorage.setItem('apiBaseUrl', url);
}

let statusCheckInterval = null;
let isLoading = false;

// Dataset configuration
const DATASET_CONFIG = {
    msmarco: {
        name: 'MS MARCO',
        defaultIndex: 'msmarco',
        endpoint: '/load/msmarco',
        showChunking: true,
        showLocale: false
    },
    esci: {
        name: 'Amazon ESCI',
        defaultIndex: 'products',
        endpoint: '/load/esci',
        showChunking: false,
        showLocale: true
    },
    'product-search': {
        name: 'Product Search Corpus',
        defaultIndex: 'product-corpus',
        endpoint: '/load/product-search',
        showChunking: false,
        showLocale: false
    },
    'open-food-facts': {
        name: 'Open Food Facts',
        defaultIndex: 'food-products',
        endpoint: '/load/open-food-facts',
        showChunking: false,
        showLocale: false
    },
    'amazon-best-sellers': {
        name: 'Amazon Best Sellers',
        defaultIndex: 'amazon-best-sellers',
        endpoint: '/load/amazon-best-sellers',
        showChunking: false,
        showLocale: false
    }
};

// Handle dataset selection change
function onDatasetChange() {
    const dataset = document.getElementById('dataset-select').value;
    const config = DATASET_CONFIG[dataset];
    
    // Safety check - if dataset not found, default to msmarco
    if (!config) {
        console.warn('Unknown dataset:', dataset);
        return;
    }
    
    // Update index name placeholder
    document.getElementById('index-name').value = config.defaultIndex;
    
    // Show/hide chunking options
    const chunkRow = document.querySelector('.chunk-options');
    if (chunkRow) {
        chunkRow.style.display = config.showChunking ? 'flex' : 'none';
    }
    
    // Show/hide locale selector
    const localeGroup = document.getElementById('locale-group');
    if (localeGroup) {
        localeGroup.style.display = config.showLocale ? 'block' : 'none';
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Load saved API URL into input field
    const apiUrlInput = document.getElementById('api-url');
    if (apiUrlInput) {
        apiUrlInput.value = getApiBaseUrl();
    }
    
    // Initialize dataset selector
    onDatasetChange();
    
    checkHealth();
    loadIndices();
    
    // Check health every 30 seconds
    setInterval(checkHealth, 30000);
});

// Update API URL from settings
function updateApiUrl() {
    let newUrl = document.getElementById('api-url').value.trim();
    if (newUrl) {
        // Ensure URL has protocol
        if (!newUrl.startsWith('http://') && !newUrl.startsWith('https://')) {
            newUrl = 'http://' + newUrl;
            document.getElementById('api-url').value = newUrl;
        }
        setApiBaseUrl(newUrl);
        showToast('API URL updated! Reconnecting...', 'success');
        checkHealth();
        loadIndices();
    } else {
        showToast('Please enter a valid URL', 'error');
    }
}

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
        const response = await fetch(`${getApiBaseUrl()}${endpoint}`, {
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

// Load dataset (MS MARCO or ESCI)
async function loadData() {
    if (isLoading) {
        showToast('A loading operation is already in progress', 'warning');
        return;
    }
    
    const dataset = document.getElementById('dataset-select').value;
    const datasetConfig = DATASET_CONFIG[dataset];
    const indexName = document.getElementById('index-name').value || datasetConfig.defaultIndex;
    const maxDocs = parseInt(document.getElementById('max-docs').value) || null;
    const batchSize = parseInt(document.getElementById('batch-size').value) || 500;
    const embeddingBatch = parseInt(document.getElementById('embedding-batch').value) || 32;
    
    const loadBtn = document.getElementById('load-btn');
    const progressContainer = document.getElementById('progress-container');
    
    try {
        loadBtn.disabled = true;
        loadBtn.innerHTML = '⏳ Starting...';
        progressContainer.style.display = 'block';
        
        let config = {
            index_name: indexName,
            batch_size: batchSize,
            embedding_batch_size: embeddingBatch
        };
        
        if (maxDocs && maxDocs > 0) {
            config.max_documents = maxDocs;
        }
        
        // Add dataset-specific config
        if (dataset === 'msmarco') {
            config.dataset_split = 'train';
            config.chunk_size = parseInt(document.getElementById('chunk-size').value) || 512;
            config.chunk_overlap = parseInt(document.getElementById('chunk-overlap').value) || 50;
        } else if (dataset === 'esci') {
            config.locale = document.getElementById('locale-select').value || 'us';
        }
        
        await apiCall(datasetConfig.endpoint, {
            method: 'POST',
            body: JSON.stringify(config)
        });
        
        isLoading = true;
        showToast(`${datasetConfig.name} data loading started!`, 'success');
        
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
                ${result.hits.map(hit => {
                    const source = hit.source;
                    // Get display text from various possible fields
                    const displayText = source.text || source.passage_text || source.product_title || 
                                       source.product_name || source.name || source.title || 
                                       source.combined_text || 'No title available';
                    // Get optional description/details
                    const description = source.description || source.product_description || '';
                    // Get category if available
                    const category = source.category || source.product_category || '';
                    // Get price if available
                    const price = source.product_price || source.price || '';
                    // Get rating if available
                    const rating = source.product_star_rating || source.rating || '';
                    
                    return `
                    <div class="search-result-item">
                        <span class="score">Score: ${hit.score.toFixed(4)}</span>
                        <p class="passage">${escapeHtml(displayText)}</p>
                        ${description ? `<p class="description" style="color: var(--text-secondary); font-size: 0.9em; margin-top: 5px;">${escapeHtml(description.substring(0, 200))}${description.length > 200 ? '...' : ''}</p>` : ''}
                        ${category ? `<span class="category" style="background: var(--accent-color); color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75em; margin-right: 8px;">${escapeHtml(category)}</span>` : ''}
                        ${price ? `<span class="price" style="color: var(--success-color); font-weight: bold;">$${escapeHtml(price)}</span>` : ''}
                        ${rating ? `<span class="rating" style="margin-left: 8px;">⭐ ${rating}</span>` : ''}
                        ${source.query_text ? `<p class="query">Query: ${escapeHtml(source.query_text)}</p>` : ''}
                        ${source.chunk_index !== undefined ? `<span class="chunk-info">Chunk ${source.chunk_index}</span>` : ''}
                    </div>
                `}).join('')}
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
