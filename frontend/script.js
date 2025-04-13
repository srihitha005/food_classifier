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
  fetch('http://localhost:5000/predict', {
      method: 'POST',
      body: formData
  })
  .then(response => response.json())
  .then(data => {
      // Display the prediction result or error
      if (data.error) {
          resultDiv.innerHTML = `<p>Error: ${data.error}</p>`;
      } else {
          resultDiv.innerHTML = `<p>The fruit/vegetable is: <strong>${data.prediction}</strong></p>`;
      }
  })
  .catch(error => {
      resultDiv.innerHTML = `<p>Error: ${error.message}</p>`;
  });
}