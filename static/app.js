// API Base URL
const API_BASE = '/api';

// Step state management
const stepState = {
    currentStep: 'translate',
    completedSteps: new Set(),
    formData: {}
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeStepper();
    loadModels();
    loadFormData();
});

// Initialize stepper navigation
function initializeStepper() {
    const stepperSegments = document.querySelectorAll('.stepper-segment');
    const stepContents = document.querySelectorAll('.step-content');
    
    stepperSegments.forEach(segment => {
        segment.addEventListener('click', () => {
            const targetStep = segment.getAttribute('data-step');
            switchStep(targetStep);
        });
        
        // Keyboard navigation
        segment.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                const targetStep = segment.getAttribute('data-step');
                switchStep(targetStep);
            }
        });
    });
    
    // Update stepper visual state
    updateStepperState();
}

// Switch between steps
function switchStep(stepName) {
    // Save current form data
    saveFormData();
    
    // Update current step
    stepState.currentStep = stepName;
    
    // Update stepper segments
    const stepperSegments = document.querySelectorAll('.stepper-segment');
    stepperSegments.forEach(segment => {
        const step = segment.getAttribute('data-step');
        segment.classList.remove('active');
        if (step === stepName) {
            segment.classList.add('active');
            segment.setAttribute('aria-current', 'step');
        } else {
            segment.removeAttribute('aria-current');
        }
    });
    
    // Update step content
    const stepContents = document.querySelectorAll('.step-content');
    stepContents.forEach(content => {
        content.classList.remove('active');
    });
    
    const targetContent = document.getElementById(`${stepName}-step`);
    if (targetContent) {
        targetContent.classList.add('active');
    }
    
    // Restore form data for new step
    restoreFormData(stepName);
    
    // Update stepper visual state
    updateStepperState();
}

// Update stepper visual state (active/completed/inactive)
function updateStepperState() {
    const stepperSegments = document.querySelectorAll('.stepper-segment');
    const stepOrder = ['translate', 'rtl', 'nodes', 'complete'];
    const currentIndex = stepOrder.indexOf(stepState.currentStep);
    
    stepperSegments.forEach((segment, index) => {
        const step = segment.getAttribute('data-step');
        segment.classList.remove('completed', 'active');
        
        if (step === stepState.currentStep) {
            segment.classList.add('active');
        } else if (stepState.completedSteps.has(step) || index < currentIndex) {
            segment.classList.add('completed');
        }
    });
}

// Save form data for current step
function saveFormData() {
    const currentStep = stepState.currentStep;
    const stepData = {};
    
    // Get all inputs, selects, and textareas in current step
    const currentStepElement = document.getElementById(`${currentStep}-step`);
    if (currentStepElement) {
        const inputs = currentStepElement.querySelectorAll('input, select, textarea');
        inputs.forEach(input => {
            if (input.type === 'checkbox') {
                stepData[input.id] = input.checked;
            } else {
                stepData[input.id] = input.value;
            }
        });
    }
    
    stepState.formData[currentStep] = stepData;
}

// Restore form data for a step
function restoreFormData(stepName) {
    const stepData = stepState.formData[stepName];
    if (!stepData) return;
    
    Object.keys(stepData).forEach(inputId => {
        const input = document.getElementById(inputId);
        if (input) {
            if (input.type === 'checkbox') {
                input.checked = stepData[inputId];
            } else {
                input.value = stepData[inputId];
            }
        }
    });
}

// Load form data from sessionStorage on page load
function loadFormData() {
    try {
        const saved = sessionStorage.getItem('workflowConverterFormData');
        if (saved) {
            stepState.formData = JSON.parse(saved);
            // Restore data for current step
            restoreFormData(stepState.currentStep);
        }
    } catch (e) {
        console.error('Error loading form data:', e);
    }
}

// Save form data to sessionStorage
function persistFormData() {
    try {
        sessionStorage.setItem('workflowConverterFormData', JSON.stringify(stepState.formData));
    } catch (e) {
        console.error('Error saving form data:', e);
    }
}

// Mark step as completed
function markStepCompleted(stepName) {
    stepState.completedSteps.add(stepName);
    updateStepperState();
}

// Load available models
async function loadModels() {
    try {
        const response = await fetch(`${API_BASE}/models`);
        const data = await response.json();
        
        const modelSelects = ['trans-model', 'nodes-model', 'complete-model'];
        modelSelects.forEach(selectId => {
            const select = document.getElementById(selectId);
            if (select) {
                select.innerHTML = '';
                Object.entries(data.models).forEach(([key, model]) => {
                    const option = document.createElement('option');
                    option.value = key;
                    option.textContent = `${model.name} - $${model.input_cost}/${model.output_cost}`;
                    if (key === data.default) {
                        option.selected = true;
                    }
                    select.appendChild(option);
                });
            }
        });
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

// Analyze workflow
async function analyzeWorkflow(type) {
    const workflowText = document.getElementById(`${type}-workflow`).value;
    
    if (!workflowText.trim()) {
        showError(`${type}-result`, 'Please paste workflow JSON first');
        return;
    }
    
    try {
        showLoading(`${type}-result`);
        
        const response = await fetch(`${API_BASE}/workflow/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                workflow_json: workflowText
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showInfo(`${type}-result`, `
                <h3>Workflow Analysis</h3>
                <div class="info-box success">
                    <strong>Version:</strong> ${data.version}<br>
                    <strong>Supported:</strong> ${data.is_supported ? 'Yes' : 'No'}<br>
                    <strong>Total Nodes:</strong> ${data.total_nodes}<br>
                    <strong>Sticky Notes:</strong> ${data.sticky_notes_count}<br>
                    <strong>Has RTL Content:</strong> ${data.has_rtl_content ? 'Yes' : 'No'}<br>
                    <strong>RTL Positioned:</strong> ${data.is_rtl_positioned ? 'Yes' : 'No'}
                </div>
            `);
        } else {
            showError(`${type}-result`, data.detail || 'Analysis failed');
        }
    } catch (error) {
        showError(`${type}-result`, `Error: ${error.message}`);
    }
}

// Translate workflow
async function translateWorkflow() {
    const apiKey = document.getElementById('trans-api-key').value;
    const model = document.getElementById('trans-model').value;
    const language = document.getElementById('trans-language').value;
    const workflowText = document.getElementById('trans-workflow').value;
    
    if (!apiKey) {
        showError('trans-result', 'Please enter your API key');
        return;
    }
    
    if (!workflowText.trim()) {
        showError('trans-result', 'Please paste workflow JSON');
        return;
    }
    
    try {
        showLoading('trans-result');
        
        const response = await fetch(`${API_BASE}/workflow/translate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                workflow_json: workflowText,
                api_key: apiKey,
                model: model,
                target_language: language
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            markStepCompleted('translate');
            saveFormData();
            persistFormData();
            showSuccess('trans-result', data.workflow, `translated_workflow_${language}.json`);
        } else {
            showError('trans-result', data.detail || 'Translation failed');
        }
    } catch (error) {
        showError('trans-result', `Error: ${error.message}`);
    }
}

// Convert to RTL
async function convertRTL() {
    const workflowText = document.getElementById('rtl-workflow').value;
    const canvasWidth = parseFloat(document.getElementById('rtl-canvas-width').value) || null;
    
    if (!workflowText.trim()) {
        showError('rtl-result', 'Please paste workflow JSON');
        return;
    }
    
    try {
        showLoading('rtl-result');
        
        const response = await fetch(`${API_BASE}/workflow/convert-rtl`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                workflow_json: workflowText,
                canvas_width: canvasWidth === 0 ? null : canvasWidth
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            markStepCompleted('rtl');
            saveFormData();
            persistFormData();
            showSuccess('rtl-result', data.workflow, 'rtl_workflow.json');
        } else {
            showError('rtl-result', data.detail || 'RTL conversion failed');
        }
    } catch (error) {
        showError('rtl-result', `Error: ${error.message}`);
    }
}

// Analyze node names
async function analyzeNodeNames() {
    const workflowText = document.getElementById('nodes-workflow').value;
    
    if (!workflowText.trim()) {
        showError('nodes-result', 'Please paste workflow JSON first');
        return;
    }
    
    try {
        showLoading('nodes-result');
        
        const response = await fetch(`${API_BASE}/workflow/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                workflow_json: workflowText
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showInfo('nodes-result', `
                <h3>Node Analysis</h3>
                <div class="info-box info">
                    <strong>Total Nodes:</strong> ${data.total_nodes}<br>
                    Ready for node name translation
                </div>
            `);
        } else {
            showError('nodes-result', data.detail || 'Analysis failed');
        }
    } catch (error) {
        showError('nodes-result', `Error: ${error.message}`);
    }
}

// Translate node names
async function translateNodeNames() {
    const apiKey = document.getElementById('nodes-api-key').value;
    const model = document.getElementById('nodes-model').value;
    const language = document.getElementById('nodes-language').value;
    const workflowText = document.getElementById('nodes-workflow').value;
    
    if (!apiKey) {
        showError('nodes-result', 'Please enter your API key');
        return;
    }
    
    if (!workflowText.trim()) {
        showError('nodes-result', 'Please paste workflow JSON');
        return;
    }
    
    try {
        showLoading('nodes-result');
        
        const response = await fetch(`${API_BASE}/workflow/translate-nodes`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                workflow_json: workflowText,
                api_key: apiKey,
                model: model,
                target_language: language
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            markStepCompleted('nodes');
            saveFormData();
            persistFormData();
            showSuccess('nodes-result', data.workflow, `persian_node_names_${model}.json`);
        } else {
            showError('nodes-result', data.detail || 'Translation failed');
        }
    } catch (error) {
        showError('nodes-result', `Error: ${error.message}`);
    }
}

// Complete localization
async function completeLocalization() {
    const apiKey = document.getElementById('complete-api-key').value;
    const model = document.getElementById('complete-model').value;
    const language = document.getElementById('complete-language').value;
    const workflowText = document.getElementById('complete-workflow').value;
    const translateNotes = document.getElementById('complete-translate-notes').checked;
    const translateNodes = document.getElementById('complete-translate-nodes').checked;
    const convertRTL = document.getElementById('complete-convert-rtl').checked;
    const canvasWidth = parseFloat(document.getElementById('complete-canvas-width').value) || null;
    
    if (!apiKey) {
        showError('complete-result', 'Please enter your API key');
        return;
    }
    
    if (!workflowText.trim()) {
        showError('complete-result', 'Please paste workflow JSON');
        return;
    }
    
    if (!translateNotes && !translateNodes && !convertRTL) {
        showError('complete-result', 'Please select at least one processing option');
        return;
    }
    
    try {
        showLoading('complete-result');
        
        const response = await fetch(`${API_BASE}/workflow/complete-localization`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                workflow_json: workflowText,
                api_key: apiKey,
                model: model,
                target_language: language,
                translate_notes: translateNotes,
                translate_node_names: translateNodes,
                convert_to_rtl: convertRTL,
                canvas_width: canvasWidth === 0 ? null : canvasWidth
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            markStepCompleted('complete');
            saveFormData();
            persistFormData();
            
            const filenameParts = [];
            if (data.results.translate_notes) filenameParts.push('translated');
            if (data.results.translate_nodes) filenameParts.push('persian_names');
            if (data.results.convert_rtl) filenameParts.push('rtl');
            const filename = `localized_workflow_${filenameParts.join('_')}_${model}.json`;
            
            showSuccess('complete-result', data.workflow, filename);
        } else {
            showError('complete-result', data.detail || 'Localization failed');
        }
    } catch (error) {
        showError('complete-result', `Error: ${error.message}`);
    }
}

// UI Helper functions
function showLoading(resultId) {
    const resultArea = document.getElementById(resultId);
    resultArea.innerHTML = '<div class="loading"></div> Processing...';
    resultArea.classList.add('active');
}

function showSuccess(resultId, workflow, filename) {
    const resultArea = document.getElementById(resultId);
    const jsonStr = JSON.stringify(workflow, null, 2);
    
    // Store workflow data for download
    const downloadId = `download-${Date.now()}`;
    window[downloadId] = workflow;
    
    resultArea.innerHTML = `
        <h3>Success</h3>
        <div class="info-box success">Workflow processed successfully</div>
        <pre>${escapeHtml(jsonStr)}</pre>
        <button class="download-btn" onclick="downloadJSON(window['${downloadId}'], '${filename}')" type="button">
            Download Workflow
        </button>
    `;
    resultArea.classList.add('active');
}

function showError(resultId, message) {
    const resultArea = document.getElementById(resultId);
    resultArea.innerHTML = `
        <div class="info-box error">
            <strong>Error:</strong> ${escapeHtml(message)}
        </div>
    `;
    resultArea.classList.add('active');
}

function showInfo(resultId, html) {
    const resultArea = document.getElementById(resultId);
    resultArea.innerHTML = html;
    resultArea.classList.add('active');
}

function downloadJSON(jsonData, filename) {
    // Parse if string, stringify if object
    let dataStr;
    if (typeof jsonData === 'string') {
        try {
            // If it's already a JSON string, use it directly
            JSON.parse(jsonData);
            dataStr = jsonData;
        } catch {
            // If not valid JSON, stringify it
            dataStr = JSON.stringify(jsonData, null, 2);
        }
    } else {
        dataStr = JSON.stringify(jsonData, null, 2);
    }
    
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Auto-save form data periodically
setInterval(() => {
    saveFormData();
    persistFormData();
}, 5000);

// Save form data before page unload
window.addEventListener('beforeunload', () => {
    saveFormData();
    persistFormData();
});
