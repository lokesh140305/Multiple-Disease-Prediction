// Tab switching functionality
document.addEventListener('DOMContentLoaded', function() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.getAttribute('data-tab');
            
            // Remove active class from all buttons and contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to clicked button and corresponding content
            button.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
        });
    });

    // Form submissions
    const diabetesForm = document.getElementById('diabetes-form');
    const heartForm = document.getElementById('heart-form');
    const parkinsonsForm = document.getElementById('parkinsons-form');

    diabetesForm.addEventListener('submit', (e) => {
        e.preventDefault();
        predictDisease('diabetes', diabetesForm);
    });

    heartForm.addEventListener('submit', (e) => {
        e.preventDefault();
        predictDisease('heart', heartForm);
    });

    parkinsonsForm.addEventListener('submit', (e) => {
        e.preventDefault();
        predictDisease('parkinsons', parkinsonsForm);
    });
});

function predictDisease(diseaseType, form) {
    const resultDiv = document.getElementById(`${diseaseType}-result`);
    resultDiv.innerHTML = '<div class="loading">🔄 Processing your data...</div>';
    resultDiv.classList.add('show');
    resultDiv.classList.remove('low-risk', 'high-risk', 'error');

    // Collect form data
    const formData = new FormData(form);
    const data = {};
    
    if (diseaseType === 'diabetes') {
        data.diabetes = {
            pregnancies: formData.get('pregnancies'),
            glucose: formData.get('glucose'),
            blood_pressure: formData.get('blood_pressure'),
            skin_thickness: formData.get('skin_thickness'),
            insulin: formData.get('insulin'),
            bmi: formData.get('bmi'),
            diabetes_pedigree: formData.get('diabetes_pedigree'),
            age: formData.get('age')
        };
    } else if (diseaseType === 'heart') {
        data.heart = {
            age: formData.get('age'),
            sex: formData.get('sex'),
            cp: formData.get('cp'),
            trestbps: formData.get('trestbps'),
            chol: formData.get('chol'),
            fbs: formData.get('fbs'),
            restecg: formData.get('restecg'),
            thalach: formData.get('thalach'),
            exang: formData.get('exang'),
            oldpeak: formData.get('oldpeak'),
            slope: formData.get('slope'),
            ca: formData.get('ca'),
            thal: formData.get('thal')
        };
    } else if (diseaseType === 'parkinsons') {
        data.parkinsons = {
            jitter: formData.get('jitter'),
            shimmer: formData.get('shimmer'),
            nhr: formData.get('nhr'),
            hnr: formData.get('hnr'),
            rpde: formData.get('rpde'),
            dfa: formData.get('dfa'),
            spread1: formData.get('spread1'),
            spread2: formData.get('spread2'),
            d2: formData.get('d2'),
            ppe: formData.get('ppe')
        };
    }

    // Send prediction request
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        if (result.error) {
            throw new Error(result.error);
        }

        const prediction = result[diseaseType];
        if (!prediction) {
            throw new Error('No prediction result received');
        }

        const riskLevel = prediction.risk;
        const probability = prediction.probability.toFixed(2);
        
        // Update result display
        resultDiv.classList.remove('low-risk', 'high-risk', 'error');
        resultDiv.classList.add(riskLevel === 'High' ? 'high-risk' : 'low-risk');
        
        const diseaseNames = {
            'diabetes': 'Diabetes',
            'heart': 'Heart Disease',
            'parkinsons': "Parkinson's Disease"
        };
        
        resultDiv.innerHTML = `
            <h3>${diseaseNames[diseaseType]} Risk Assessment</h3>
            <p><strong>Risk Level:</strong> <span style="font-size: 1.3em;">${riskLevel} Risk</span></p>
            <p><strong>Probability:</strong> ${probability}%</p>
            <div class="probability-bar">
                <div class="probability-fill" style="width: ${probability}%">${probability}%</div>
            </div>
            <p style="margin-top: 15px; font-size: 0.95em; opacity: 0.8;">
                ${riskLevel === 'High' 
                    ? '⚠️ Please consult with a healthcare professional for further evaluation.' 
                    : '✅ Your risk appears to be low. Continue maintaining a healthy lifestyle!'}
            </p>
        `;
    })
    .catch(error => {
        resultDiv.classList.remove('low-risk', 'high-risk');
        resultDiv.classList.add('error');
        resultDiv.innerHTML = `
            <h3>Error</h3>
            <p>${error.message || 'An error occurred while processing your request. Please try again.'}</p>
        `;
    });
}

