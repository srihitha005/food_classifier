function uploadImage() {
    // Get the file input and result display elements
    const fileInput = document.getElementById('imageUpload');
    const resultDiv = document.getElementById('result');
    const file = fileInput.files[0];
  
    // Check if a file is selected
    if (!file) {
        resultDiv.innerHTML = '<p>Please select an image first.</p>';
        return;
    }
  
    // Create FormData object to send the file
    const formData = new FormData();
    formData.append('file', file);
  
    // Send the file to the backend for prediction
    fetch('https://food-classifier-fx3m.onrender.com/predict', { // Update to deployed URL
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Display the prediction result or error
        if (data.error) {
            resultDiv.innerHTML = `<p>Error: ${data.error}</p>`;
        } else {
            resultDiv.innerHTML = `<p>The fruit/vegetable is: <strong>${data.result}</strong> (Confidence: ${data.confidence.toFixed(2)})</p>`;
        }
    })
    .catch(error => {
        resultDiv.innerHTML = `<p>Error: ${error.message}</p>`;
    });
  }