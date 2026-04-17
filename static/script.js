// DOM Elements
const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const imagePreview = document.getElementById('imagePreview');
const changeImageBtn = document.getElementById('changeImageBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('resultsSection');
const analyzeAnotherBtn = document.getElementById('analyzeAnotherBtn');

let selectedFile = null;
let originalImageDataUrl = null;

// Event Listeners
uploadBox.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
changeImageBtn.addEventListener('click', resetUpload);
analyzeBtn.addEventListener('click', analyzeImage);
analyzeAnotherBtn.addEventListener('click', resetUpload);

// Drag and drop handlers
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragover');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// File handling functions
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file (JPG, PNG, JPEG)');
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        alert('File size must be less than 10MB');
        return;
    }

    selectedFile = file;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        originalImageDataUrl = e.target.result;
        imagePreview.src = e.target.result;
        uploadBox.style.display = 'none';
        previewSection.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    selectedFile = null;
    originalImageDataUrl = null;
    fileInput.value = '';
    uploadBox.style.display = 'block';
    previewSection.style.display = 'none';
    resultsSection.style.display = 'none';
    loading.style.display = 'none';
}

async function analyzeImage() {
    if (!selectedFile) {
        alert('Please select an image first');
        return;
    }

    previewSection.style.display = 'none';
    loading.style.display = 'block';
    resultsSection.style.display = 'none';

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Analysis failed');
        }

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        console.error('Error:', error);
        alert(`Error analyzing image: ${error.message}`);
        previewSection.style.display = 'block';
    } finally {
        loading.style.display = 'none';
    }
}

function displayResults(data) {
    resultsSection.style.display = 'block';

    // Predicted class
    const predictedClass = document.getElementById('predictedClass');
    predictedClass.textContent = data.predicted_class;
    
    predictedClass.className = 'result-value';
    if (data.is_fish === false) {
        predictedClass.classList.add('warning');
    } else if (data.predicted_class.toLowerCase().includes('healthy')) {
        predictedClass.classList.add('healthy');
    } else if (data.confidence < 0.7) {
        predictedClass.classList.add('warning');
    } else {
        predictedClass.classList.add('danger');
    }

    // Confidence
    const confidenceValue = document.getElementById('confidenceValue');
    confidenceValue.textContent = data.confidence_percentage;
    
    const confidenceBar = document.getElementById('confidenceBar');
    confidenceBar.style.width = `${data.confidence * 100}%`;

    // Confidence badge
    const confidenceBadge = document.getElementById('confidenceBadge');
    if (data.confidence >= 0.9) {
        confidenceBadge.textContent = 'Very High';
        confidenceBadge.className = 'confidence-badge badge-high';
    } else if (data.confidence >= 0.7) {
        confidenceBadge.textContent = 'High';
        confidenceBadge.className = 'confidence-badge badge-high';
    } else if (data.confidence >= 0.5) {
        confidenceBadge.textContent = 'Moderate';
        confidenceBadge.className = 'confidence-badge badge-moderate';
    } else {
        confidenceBadge.textContent = 'Low';
        confidenceBadge.className = 'confidence-badge badge-low';
    }

    // Diagnosis message
    const diagnosisMessage = document.getElementById('diagnosisMessage');
    const messageBox = document.getElementById('messageBox');
    diagnosisMessage.textContent = data.message;
    
    messageBox.className = 'message-box';
    if (data.is_fish === false) {
        messageBox.classList.add('message-warning');
    } else if (data.predicted_class.toLowerCase().includes('healthy')) {
        messageBox.classList.add('message-success');
    } else {
        messageBox.classList.add('message-danger');
    }

    // All predictions
    const predictionsList = document.getElementById('predictionsList');
    predictionsList.innerHTML = '';
    
    data.all_predictions.forEach(prediction => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        
        const percentage = (prediction.confidence * 100).toFixed(2);
        const isTop = prediction.class === data.predicted_class;
        
        item.innerHTML = `
            <span class="prediction-name ${isTop ? 'prediction-top' : ''}">${prediction.class}</span>
            <div class="prediction-bar-container">
                <div class="prediction-bar-fill ${isTop ? 'prediction-bar-top' : ''}" style="width: ${percentage}%"></div>
            </div>
            <span class="prediction-confidence ${isTop ? 'prediction-top' : ''}">${percentage}%</span>
        `;
        
        predictionsList.appendChild(item);
    });

    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Health check on load
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        if (!data.model_loaded) {
            console.warn('Model not loaded properly');
        }
        
        console.log('Health check:', data);
        
        // Show version info if available
        if (data.version) {
            console.log(`API Version: ${data.version}`);
        }
    } catch (error) {
        console.error('Health check failed:', error);
    }
});
