// Application State
let currentTab = "train";

// Initialize app
document.addEventListener("DOMContentLoaded", function () {
  initializeTabs();
  initializeForms();
  checkModelStatus();
});

// Tab Management
function initializeTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetTab = btn.dataset.tab;

      // Update active states
      tabBtns.forEach((b) => b.classList.remove("active"));
      tabPanes.forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      document.getElementById(targetTab).classList.add("active");

      currentTab = targetTab;
    });
  });
}

// Form Initialization
function initializeForms() {
  // Train form
  document.getElementById("trainForm").addEventListener("submit", handleTrain);

  // Predict forms
  document
    .getElementById("singlePredictForm")
    .addEventListener("submit", handleSinglePredict);
  document
    .getElementById("batchPredictForm")
    .addEventListener("submit", handleBatchPredict);

  // Predict options
  const optionBtns = document.querySelectorAll(".option-btn");
  optionBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      optionBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      const option = btn.dataset.option;
      document.getElementById("singleForm").style.display =
        option === "single" ? "block" : "none";
      document.getElementById("batchForm").style.display =
        option === "batch" ? "block" : "none";
    });
  });

  // Visualization button
  document
    .getElementById("generateVizBtn")
    .addEventListener("click", handleVisualize);

  // Stats button
  document
    .getElementById("refreshStatsBtn")
    .addEventListener("click", handleStats);
}

// Check model status
async function checkModelStatus() {
  try {
    const response = await fetch("/api/stats");
    const data = await response.json();

    if (data.success) {
      document.getElementById("modelStatus").textContent = "Trained";
      document.getElementById("modelStatus").style.color = "#48bb78";
    }
  } catch (error) {
    document.getElementById("modelStatus").textContent = "Not Trained";
    document.getElementById("modelStatus").style.color = "#f56565";
  }
}

// Train Model
async function handleTrain(e) {
  e.preventDefault();

  const btn = document.getElementById("trainBtn");
  const btnText = btn.querySelector(".btn-text");
  const btnLoader = btn.querySelector(".btn-loader");

  // Get form values
  const nSamples = parseInt(document.getElementById("nSamples").value);
  const anomalyRatio =
    parseFloat(document.getElementById("anomalyRatio").value) / 100;

  // Disable button and show loader
  btn.disabled = true;
  btnText.textContent = "Training...";
  btnLoader.style.display = "block";

  try {
    const response = await fetch("/api/train", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        n_samples: nSamples,
        anomaly_ratio: anomalyRatio,
      }),
    });

    const data = await response.json();

    if (data.success) {
      showToast("Model trained successfully!", "success");
      displayTrainResults(data);
      checkModelStatus();
    } else {
      showToast(data.error || "Training failed", "error");
    }
  } catch (error) {
    showToast("Network error: " + error.message, "error");
  } finally {
    btn.disabled = false;
    btnText.textContent = "Train Model";
    btnLoader.style.display = "none";
  }
}

// Display train results
function displayTrainResults(data) {
  const resultsBox = document.getElementById("trainResults");
  resultsBox.style.display = "block";

  resultsBox.innerHTML = `
        <h3 style="margin-bottom: 16px; font-size: 18px; font-weight: 600;">Training Results</h3>
        <div class="result-item">
            <div class="result-header">
                <span class="result-label">Samples Generated</span>
                <span style="font-weight: 600; color: #667eea;">${data.samples_generated.toLocaleString()}</span>
            </div>
        </div>
        <div class="result-item">
            <div class="result-header">
                <span class="result-label">Anomalies Detected</span>
                <span style="font-weight: 600; color: #f56565;">${data.anomalies_detected}</span>
            </div>
        </div>
        <div class="result-item">
            <div class="result-header">
                <span class="result-label">Accuracy</span>
                <span style="font-weight: 600; color: #48bb78;">${(data.metrics.accuracy * 100).toFixed(2)}%</span>
            </div>
        </div>
        <div class="result-item">
            <div class="result-header">
                <span class="result-label">Precision</span>
                <span style="font-weight: 600;">${(data.metrics.precision * 100).toFixed(2)}%</span>
            </div>
        </div>
        <div class="result-item">
            <div class="result-header">
                <span class="result-label">Recall</span>
                <span style="font-weight: 600;">${(data.metrics.recall * 100).toFixed(2)}%</span>
            </div>
        </div>
        <div class="result-item">
            <div class="result-header">
                <span class="result-label">F1 Score</span>
                <span style="font-weight: 600;">${(data.metrics.f1_score * 100).toFixed(2)}%</span>
            </div>
        </div>
    `;
}

// Single Predict
async function handleSinglePredict(e) {
  e.preventDefault();

  const btn = document.getElementById("predictBtn");
  const btnText = btn.querySelector(".btn-text");
  const btnLoader = btn.querySelector(".btn-loader");

  // Get form values
  const formData = {
    login_hour: parseInt(document.getElementById("loginHour").value),
    login_attempts: parseInt(document.getElementById("loginAttempts").value),
    ip_frequency: parseInt(document.getElementById("ipFrequency").value),
    device_type: parseInt(document.getElementById("deviceType").value),
    login_success: parseInt(document.getElementById("loginSuccess").value),
  };

  btn.disabled = true;
  btnText.textContent = "Analyzing...";
  btnLoader.style.display = "block";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formData),
    });

    const data = await response.json();

    if (data.success) {
      showToast("Analysis complete!", "success");
      displayPredictResults(data);
    } else {
      showToast(data.error || "Prediction failed", "error");
    }
  } catch (error) {
    showToast("Network error: " + error.message, "error");
  } finally {
    btn.disabled = false;
    btnText.textContent = "Analyze Login";
    btnLoader.style.display = "none";
  }
}

// Batch Predict
async function handleBatchPredict(e) {
  e.preventDefault();

  const btn = document.getElementById("batchPredictBtn");
  const btnText = btn.querySelector(".btn-text");
  const btnLoader = btn.querySelector(".btn-loader");
  const fileInput = document.getElementById("csvFile");

  if (!fileInput.files.length) {
    showToast("Please select a CSV file", "error");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  btn.disabled = true;
  btnText.textContent = "Analyzing...";
  btnLoader.style.display = "block";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.success) {
      showToast("Batch analysis complete!", "success");
      displayPredictResults(data);
    } else {
      showToast(data.error || "Prediction failed", "error");
    }
  } catch (error) {
    showToast("Network error: " + error.message, "error");
  } finally {
    btn.disabled = false;
    btnText.textContent = "Analyze Batch";
    btnLoader.style.display = "none";
  }
}

// Display predict results
function displayPredictResults(data) {
  const resultsBox = document.getElementById("predictResults");
  resultsBox.style.display = "block";

  let html = `
        <h3 style="margin-bottom: 16px; font-size: 18px; font-weight: 600;">Analysis Results</h3>
        <div class="result-item">
            <div class="result-header">
                <span class="result-label">Total Records</span>
                <span style="font-weight: 600;">${data.total_records}</span>
            </div>
        </div>
        <div class="result-item">
            <div class="result-header">
                <span class="result-label">Normal Logins</span>
                <span style="font-weight: 600; color: #48bb78;">${data.normal_logins}</span>
            </div>
        </div>
        <div class="result-item">
            <div class="result-header">
                <span class="result-label">Anomalies Detected</span>
                <span style="font-weight: 600; color: #f56565;">${data.anomalies_detected}</span>
            </div>
        </div>
    `;

  // Show individual results (limit to first 10)
  const resultsToShow = data.results.slice(0, 10);
  if (resultsToShow.length > 0) {
    html +=
      '<h4 style="margin-top: 24px; margin-bottom: 12px; font-size: 16px; font-weight: 600;">Individual Results</h4>';

    resultsToShow.forEach((result, index) => {
      const badgeClass =
        result.prediction === "Anomaly" ? "badge-danger" : "badge-success";
      const riskBadgeClass = getRiskBadgeClass(result.risk_level);

      html += `
                <div class="result-item">
                    <div class="result-header">
                        <span class="result-label">Record #${index + 1}</span>
                        <div style="display: flex; gap: 8px;">
                            <span class="badge ${badgeClass}">${result.prediction}</span>
                            <span class="badge ${riskBadgeClass}">${result.risk_level}</span>
                        </div>
                    </div>
                    <div style="font-size: 12px; color: #718096; margin-top: 8px;">
                        Score: ${result.anomaly_score.toFixed(4)} | 
                        Hour: ${result.features.login_hour} | 
                        Attempts: ${result.features.login_attempts} | 
                        IP Freq: ${result.features.ip_frequency}
                    </div>
                </div>
            `;
    });

    if (data.results.length > 10) {
      html += `<p style="text-align: center; color: #718096; margin-top: 12px; font-size: 14px;">Showing first 10 of ${data.results.length} results</p>`;
    }
  }

  resultsBox.innerHTML = html;
}

// Get risk badge class
function getRiskBadgeClass(riskLevel) {
  switch (riskLevel) {
    case "Critical":
      return "badge-danger";
    case "High":
      return "badge-warning";
    case "Medium":
      return "badge-info";
    default:
      return "badge-success";
  }
}

// Visualize
async function handleVisualize() {
  const btn = document.getElementById("generateVizBtn");
  const btnText = btn.querySelector(".btn-text");
  const btnLoader = btn.querySelector(".btn-loader");

  btn.disabled = true;
  btnText.textContent = "Generating...";
  btnLoader.style.display = "block";

  try {
    const response = await fetch("/api/visualize", {
      method: "POST",
    });

    const data = await response.json();

    if (data.success) {
      showToast("Visualizations generated!", "success");
      displayVisualization(data.image);
    } else {
      showToast(data.error || "Visualization failed", "error");
    }
  } catch (error) {
    showToast("Network error: " + error.message, "error");
  } finally {
    btn.disabled = false;
    btnText.textContent = "Generate Visualizations";
    btnLoader.style.display = "none";
  }
}

// Display visualization
function displayVisualization(imageData) {
  const container = document.getElementById("visualizationResults");
  container.style.display = "block";
  container.innerHTML = `<img src="${imageData}" alt="Anomaly Detection Dashboard">`;
}

// Stats
async function handleStats() {
  const btn = document.getElementById("refreshStatsBtn");
  const btnText = btn.querySelector(".btn-text");
  const btnLoader = btn.querySelector(".btn-loader");

  btn.disabled = true;
  btnText.textContent = "Loading...";
  btnLoader.style.display = "block";

  try {
    const response = await fetch("/api/stats");
    const data = await response.json();

    if (data.success) {
      showToast("Statistics updated!", "success");
      displayStats(data);
    } else {
      showToast(data.error || "Failed to load statistics", "error");
    }
  } catch (error) {
    showToast("Network error: " + error.message, "error");
  } finally {
    btn.disabled = false;
    btnText.textContent = "Refresh Statistics";
    btnLoader.style.display = "none";
  }
}

// Display stats
function displayStats(data) {
  const container = document.getElementById("statsResults");
  container.style.display = "grid";

  container.innerHTML = `
        <div class="stat-card">
            <h3>Total Records</h3>
            <p>${data.total_records.toLocaleString()}</p>
        </div>
        <div class="stat-card">
            <h3>Normal Logins</h3>
            <p>${data.normal_count.toLocaleString()}</p>
        </div>
        <div class="stat-card">
            <h3>Anomalies</h3>
            <p>${data.anomaly_count.toLocaleString()}</p>
        </div>
        <div class="stat-card">
            <h3>Anomaly Rate</h3>
            <p>${data.anomaly_percentage.toFixed(2)}%</p>
        </div>
        <div class="stat-card">
            <h3>Model Contamination</h3>
            <p>${(data.model_contamination * 100).toFixed(1)}%</p>
        </div>
        <div class="stat-card">
            <h3>Trees in Forest</h3>
            <p>${data.model_estimators}</p>
        </div>
    `;
}

// Toast notification
function showToast(message, type = "success") {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.className = `toast ${type} show`;

  setTimeout(() => {
    toast.classList.remove("show");
  }, 3000);
}
