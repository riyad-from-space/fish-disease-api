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
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file (JPG, PNG, JPEG)');
        return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        alert('File size must be less than 10MB');
        return;
    }

    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadBox.style.display = 'none';
        previewSection.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    selectedFile = null;
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

    // Show loading, hide preview
    previewSection.style.display = 'none';
    loading.style.display = 'block';
    resultsSection.style.display = 'none';

    // Create form data
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
    // Show results section
    resultsSection.style.display = 'block';

    // Display predicted class
    const predictedClass = document.getElementById('predictedClass');
    predictedClass.textContent = data.predicted_class;
    
    // Add color coding based on disease type and fish detection
    predictedClass.className = 'result-value';
    if (data.is_fish === false) {
        predictedClass.classList.add('warning');
        predictedClass.style.color = '#f59e0b'; // Orange for non-fish
    } else if (data.predicted_class.toLowerCase().includes('healthy')) {
        predictedClass.classList.add('healthy');
    } else if (data.confidence < 0.7) {
        predictedClass.classList.add('warning');
    } else {
        predictedClass.classList.add('danger');
    }

    // Display confidence
    const confidenceValue = document.getElementById('confidenceValue');
    confidenceValue.textContent = data.confidence_percentage;
    
    const confidenceBar = document.getElementById('confidenceBar');
    confidenceBar.style.width = `${data.confidence * 100}%`;

    // Display message with special styling for non-fish detection
    const diagnosisMessage = document.getElementById('diagnosisMessage');
    const messageBox = document.getElementById('messageBox');
    diagnosisMessage.textContent = data.message;
    
    // Update message box styling based on result
    if (data.is_fish === false) {
        messageBox.style.background = '#fef3c7'; // Yellow background
        messageBox.style.borderLeft = '4px solid #f59e0b'; // Orange border
    } else {
        messageBox.style.background = '#eff6ff'; // Blue background
        messageBox.style.borderLeft = '4px solid var(--primary-color)'; // Blue border
    }

    // Display all predictions
    const predictionsList = document.getElementById('predictionsList');
    predictionsList.innerHTML = '';
    
    data.all_predictions.forEach(prediction => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        
        const percentage = (prediction.confidence * 100).toFixed(2);
        
        item.innerHTML = `
            <span class="prediction-name">${prediction.class}</span>
            <div class="prediction-bar-container">
                <div class="prediction-bar-fill" style="width: ${percentage}%"></div>
            </div>
            <span class="prediction-confidence">${percentage}%</span>
        `;
        
        predictionsList.appendChild(item);
    });

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
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
    } catch (error) {
        console.error('Health check failed:', error);
    }
});
