/**
 * Handwritten Digit Recognition Client Application
 * Handles canvas drawing (mouse, touch & stylus), stroke scaling,
 * REST API communication with Flask backend, and visualization.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const canvas = document.getElementById("digitCanvas");
  const ctx = canvas.getContext("2d");
  const canvasHint = document.getElementById("canvasHint");
  const clearBtn = document.getElementById("clearBtn");
  const predictBtn = document.getElementById("predictBtn");
  const btnSpinner = document.getElementById("btnSpinner");
  const predictBtnText = document.getElementById("predictBtnText");
  const alertBox = document.getElementById("alertBox");
  const alertMessage = document.getElementById("alertMessage");
  const strokeButtons = document.querySelectorAll(".stroke-btn");
  const themeToggleBtn = document.getElementById("themeToggleBtn");
  const themeLabel = document.getElementById("themeLabel");
  
  // Results Elements
  const predictedDigitEl = document.getElementById("predictedDigit");
  const confidenceValueEl = document.getElementById("confidenceValue");
  const inputThumbnailEl = document.getElementById("inputThumbnail");
  const thumbnailPlaceholderEl = document.getElementById("thumbnailPlaceholder");
  const probBarsList = document.getElementById("probBarsList");
  const modelStatusText = document.getElementById("modelStatusText");

  // State Variables
  let isDrawing = false;
  let isCanvasDirty = false;
  let currentStrokeWidth = 22; // Medium default
  let lastX = 0;
  let lastY = 0;

  // Initialize Canvas
  function initCanvas() {
    // Fill canvas with solid black (MNIST standard)
    ctx.fillStyle = "#090d16";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = currentStrokeWidth;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
  }

  initCanvas();

  // Stroke Width Controls
  strokeButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      strokeButtons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentStrokeWidth = parseInt(btn.dataset.width, 10);
      ctx.lineWidth = currentStrokeWidth;
    });
  });

  // Calculate mouse/touch position relative to canvas coordinate space
  function getCanvasCoords(event) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    let clientX, clientY;
    if (event.touches && event.touches.length > 0) {
      clientX = event.touches[0].clientX;
      clientY = event.touches[0].clientY;
    } else {
      clientX = event.clientX;
      clientY = event.clientY;
    }

    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY
    };
  }

  // Drawing Event Handlers
  function startDrawing(e) {
    e.preventDefault();
    isDrawing = true;
    isCanvasDirty = true;
    canvasHint.classList.add("hidden");
    hideAlert();

    const coords = getCanvasCoords(e);
    lastX = coords.x;
    lastY = coords.y;

    // Draw initial dot for click/tap
    ctx.beginPath();
    ctx.arc(lastX, lastY, currentStrokeWidth / 2, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
  }

  function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();

    const coords = getCanvasCoords(e);

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(coords.x, coords.y);
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = currentStrokeWidth;
    ctx.stroke();

    lastX = coords.x;
    lastY = coords.y;
  }

  function stopDrawing(e) {
    if (isDrawing) {
      isDrawing = false;
    }
  }

  // Attach Mouse & Touch Listeners
  canvas.addEventListener("mousedown", startDrawing);
  canvas.addEventListener("mousemove", draw);
  window.addEventListener("mouseup", stopDrawing);
  canvas.addEventListener("mouseleave", stopDrawing);

  // Touch device listeners with passive: false to suppress default scrolling
  canvas.addEventListener("touchstart", startDrawing, { passive: false });
  canvas.addEventListener("touchmove", draw, { passive: false });
  canvas.addEventListener("touchend", stopDrawing, { passive: false });
  canvas.addEventListener("touchcancel", stopDrawing, { passive: false });

  // Clear Canvas Functionality
  function clearCanvas() {
    initCanvas();
    isCanvasDirty = false;
    canvasHint.classList.remove("hidden");
    hideAlert();

    // Reset Hero
    predictedDigitEl.textContent = "\u2013";
    confidenceValueEl.textContent = "\u2013%";
    confidenceValueEl.style.color = "var(--accent-success)";

    // Reset Thumbnail
    inputThumbnailEl.src = "";
    inputThumbnailEl.classList.add("hidden");
    thumbnailPlaceholderEl.classList.remove("hidden");

    // Reset Probabilities
    probBarsList.innerHTML = `
      <div class="empty-state-notice">
        Draw a digit and click &ldquo;Predict Digit&rdquo; to view class probabilities.
      </div>
    `;
  }

  clearBtn.addEventListener("click", clearCanvas);

  // Alert Notifications
  function showAlert(msg) {
    alertMessage.textContent = msg;
    alertBox.classList.remove("hidden");
    alertBox.classList.remove("shake");
    // Trigger CSS reflow to re-play animation
    void alertBox.offsetWidth;
    alertBox.classList.add("shake");
  }

  function hideAlert() {
    alertBox.classList.add("hidden");
    alertBox.classList.remove("shake");
  }

  // Prediction Submission
  async function submitPrediction() {
    if (!isCanvasDirty) {
      showAlert("Canvas is empty. Please draw a digit from 0 to 9 before predicting.");
      return;
    }

    hideAlert();
    setLoading(true);

    try {
      const imageDataUrl = canvas.toDataURL("image/png");

      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ image: imageDataUrl })
      });

      const result = await response.json();

      if (!response.ok || !result.success) {
        throw new Error(result.error || "Prediction request failed.");
      }

      displayResults(result);

    } catch (err) {
      showAlert(err.message || "An unexpected error occurred while processing.");
    } finally {
      setLoading(false);
    }
  }

  predictBtn.addEventListener("click", submitPrediction);

  // Render Prediction Results
  function displayResults(data) {
    const pred = data.prediction;
    const conf = data.confidence;

    // 1. Update Predicted Digit Hero
    predictedDigitEl.textContent = pred;
    confidenceValueEl.textContent = `${conf.toFixed(1)}%`;

    // 2. Update 28x28 Input Preview Thumbnail
    if (data.thumbnail) {
      inputThumbnailEl.src = data.thumbnail;
      inputThumbnailEl.classList.remove("hidden");
      thumbnailPlaceholderEl.classList.add("hidden");
    }

    // 3. Render Probability Distribution Bars (Digits 0 - 9)
    const probsList = data.probabilities_list || [];
    probBarsList.innerHTML = "";

    probsList.forEach(item => {
      const digit = item.digit;
      const prob = item.probability;
      const pct = (prob * 100).toFixed(1);
      const isWinner = digit === pred;

      const row = document.createElement("div");
      row.className = `prob-item ${isWinner ? "is-winner" : ""}`;

      row.innerHTML = `
        <div class="prob-digit">${digit}</div>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="width: 0%;"></div>
        </div>
        <div class="prob-percentage">${pct}%</div>
      `;

      probBarsList.appendChild(row);

      // Trigger animated bar growth after layout injection
      requestAnimationFrame(() => {
        const fill = row.querySelector(".prob-bar-fill");
        if (fill) {
          fill.style.width = `${Math.max(1, Math.min(100, pct))}%`;
        }
      });
    });
  }

  function setLoading(isLoading) {
    if (isLoading) {
      predictBtn.classList.add("loading");
      predictBtn.disabled = true;
      predictBtnText.textContent = "Processing...";
    } else {
      predictBtn.classList.remove("loading");
      predictBtn.disabled = false;
      predictBtnText.textContent = "Predict Digit";
    }
  }

  // Theme Toggle (Light / Dark)
  const themeLightOption = document.getElementById("themeLightOption");
  const themeDarkOption = document.getElementById("themeDarkOption");

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("theme", theme);
    } catch (e) {
      // Ignore local storage errors if disabled
    }
  }

  // Load saved theme or default to dark
  const initialTheme = localStorage.getItem("theme") || "dark";
  applyTheme(initialTheme);

  if (themeToggleBtn) {
    themeToggleBtn.addEventListener("click", () => {
      const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
      const nextTheme = currentTheme === "light" ? "dark" : "light";
      applyTheme(nextTheme);
    });
  }

  if (themeLightOption) {
    themeLightOption.addEventListener("click", (e) => {
      e.stopPropagation();
      applyTheme("light");
    });
  }

  if (themeDarkOption) {
    themeDarkOption.addEventListener("click", (e) => {
      e.stopPropagation();
      applyTheme("dark");
    });
  }

  // Check Backend Model Status on Page Load
  async function checkBackendStatus() {
    try {
      const res = await fetch("/info");
      if (res.ok) {
        const info = await res.json();
        modelStatusText.textContent = `Model Ready (${info.total_parameters.toLocaleString()} Params)`;
      }
    } catch {
      modelStatusText.textContent = "Connecting to Server...";
    }
  }

  checkBackendStatus();
});

