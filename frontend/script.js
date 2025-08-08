async function fetchPrediction() {
    const symbol = document.getElementById('companySelect').value;
    const resultEl = document.getElementById('predictionResult');

    try {
        const response = await fetch(`http://127.0.0.1:8000/predict/${symbol}`);
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errorText}`);
        }

        const data = await response.json();

        resultEl.innerHTML = `
            <strong>${data.company}</strong><br>
            Predicted Price: ₹${data.predicted_price.toFixed(2)}<br>
            Actual Price: ₹${data.actual_price.toFixed(2)}
        `;
    } catch (error) {
        resultEl.innerHTML = 'Error fetching prediction.';
        console.error('Error:', error);
    }
}
