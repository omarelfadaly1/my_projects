let allModels = [];          
let selectedModelId = null; 
let selectedFile = null;     
let lastAnnotatedImage = null; 


function checkHealth() {
  fetch("/health")
    .then(function (response) { return response.json(); })
    .then(function (data) {
      const statusDot = document.getElementById("statusDot");
      const statusText = document.getElementById("statusText");

      if (data.status === "ready") {
        statusDot.classList.add("ready");
        statusText.textContent = "Ready · " + data.models_loaded + " models · " + (data.gpu ? "GPU" : "CPU");
      } else {
        statusDot.classList.add("error");
        statusText.textContent = "No models loaded";
      }
    })
    .catch(function () {
      document.getElementById("statusDot").classList.add("error");
      document.getElementById("statusText").textContent = "Server unreachable";
    });
}


function loadModels() {
  fetch("/models")
    .then(function (response) { return response.json(); })
    .then(function (models) {
      allModels = models;
      drawModelCards(models);
    })
    .catch(function () {
      document.getElementById("modelGrid").innerHTML = "<p>Could not load models. Is the server running?</p>";
    });
}

function drawModelCards(models) {
  const grid = document.getElementById("modelGrid");
  grid.innerHTML = ""; 

  if (models.length === 0) {
    grid.innerHTML = "<p>No models found yet. Run evaluate.py first.</p>";
    return;
  }


  for (let i = 0; i < models.length; i++) {
    const model = models[i];

    const card = document.createElement("div");
    card.className = "model-card";
    card.dataset.modelId = model.id;

    let tagsHtml = "";
    for (let t = 0; t < model.tags.length; t++) {
      tagsHtml += '<span class="model-tag">' + model.tags[t] + "</span>";
    }

    card.innerHTML =
      '<p class="model-card-name">' + model.display_name + "</p>" +
      '<p class="model-card-desc">' + model.description + "</p>" +
      '<div class="model-card-tags">' + tagsHtml + "</div>" +
      '<div class="model-card-metrics">' +
      "  <div><p class='model-metric-label'>mAP50</p><p class='model-metric-value'>" + model.map50 + "</p></div>" +
      "  <div><p class='model-metric-label'>Precision</p><p class='model-metric-value'>" + model.precision + "</p></div>" +
      "  <div><p class='model-metric-label'>Recall</p><p class='model-metric-value'>" + model.recall + "</p></div>" +
      "  <div><p class='model-metric-label'>FPS</p><p class='model-metric-value'>" + model.fps + "</p></div>" +
      "  <div><p class='model-metric-label'>Size</p><p class='model-metric-value'>" + model.size + " MB</p></div>" +
      "</div>" +
      '<button class="btn model-card-select-btn" type="button">' + (model.available ? "Select model" : "Unavailable") + "</button>";

    if (model.available) {
      card.addEventListener("click", function () { selectModel(model.id); });
    } else {
      card.style.opacity = "0.5";
    }

    grid.appendChild(card);
  }
}

function selectModel(modelId) {
  selectedModelId = modelId;

  let chosenModel = null;
  for (let i = 0; i < allModels.length; i++) {
    if (allModels[i].id === modelId) {
      chosenModel = allModels[i];
    }
  }

  const allCards = document.querySelectorAll(".model-card");
  for (let i = 0; i < allCards.length; i++) {
    if (allCards[i].dataset.modelId === modelId) {
      allCards[i].classList.add("selected");
    } else {
      allCards[i].classList.remove("selected");
    }
  }

  document.getElementById("selectedModelPill").hidden = false;
  document.getElementById("selectedModelName").textContent = chosenModel.display_name;

  updateRunButton();
}


function setupUpload() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");

  dropzone.addEventListener("click", function () { fileInput.click(); });

  fileInput.addEventListener("change", function () {
    if (fileInput.files.length > 0) {
      handleFile(fileInput.files[0]);
    }
  });

  dropzone.addEventListener("dragover", function (event) {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", function () {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", function (event) {
    event.preventDefault();
    dropzone.classList.remove("dragover");
    if (event.dataTransfer.files.length > 0) {
      handleFile(event.dataTransfer.files[0]);
    }
  });

  document.getElementById("confidenceSlider").addEventListener("input", function (event) {
    document.getElementById("confidenceValue").textContent = parseFloat(event.target.value).toFixed(2);
  });

  document.getElementById("clearBtn").addEventListener("click", clearUpload);
  document.getElementById("runBtn").addEventListener("click", runDetection);
}

function handleFile(file) {
  const allowedExtensions = ["jpg", "jpeg", "png", "jfif"];
const extension = file.name.split(".").pop().toLowerCase();

if (!allowedExtensions.includes(extension)) {
    alert("Please upload a JPG, JPEG, JFIF or PNG image.");
    return;
}
  selectedFile = file;

  const reader = new FileReader();
  reader.onload = function (event) {
    const previewImg = document.getElementById("previewImg");
    previewImg.src = event.target.result;
    previewImg.hidden = false;
    document.getElementById("dropzoneContent").hidden = true;
  };
  reader.readAsDataURL(file);

  document.getElementById("clearBtn").disabled = false;
  updateRunButton();
}

function clearUpload() {
  selectedFile = null;
  document.getElementById("fileInput").value = "";
  document.getElementById("previewImg").hidden = true;
  document.getElementById("dropzoneContent").hidden = false;
  document.getElementById("clearBtn").disabled = true;
  updateRunButton();
}

function updateRunButton() {
  const runBtn = document.getElementById("runBtn");
  runBtn.disabled = !(selectedFile && selectedModelId);
}


function runDetection() {
  if (!selectedFile || !selectedModelId) {
    return;
  }

  document.getElementById("loadingSection").hidden = false;
  document.getElementById("resultsSection").hidden = true;

  const confidence = document.getElementById("confidenceSlider").value;

  const formData = new FormData();
  formData.append("image", selectedFile);
  formData.append("model", selectedModelId);
  formData.append("confidence", confidence);

  fetch("/predict", { method: "POST", body: formData })
    .then(function (response) { return response.json(); })
    .then(function (data) {
      document.getElementById("loadingSection").hidden = true;

      if (!data.success) {
        alert("Detection failed: " + data.error);
        return;
      }

      showResults(data);
    })
    .catch(function () {
      document.getElementById("loadingSection").hidden = true;
      alert("Something went wrong talking to the server.");
    });
}

function showResults(data) {
  document.getElementById("resultsSection").hidden = false;

  document.getElementById("originalImg").src = document.getElementById("previewImg").src;
  document.getElementById("annotatedImg").src = data.annotated_image;
  lastAnnotatedImage = data.annotated_image;

  document.getElementById("statModel").textContent = data.model;
  document.getElementById("statInferenceTime").textContent = data.inference_ms + " ms";
  document.getElementById("statObjectCount").textContent = data.object_count;
  document.getElementById("statTopDetection").textContent = data.detections.length > 0 ? data.detections[0].class : "—";

  drawDetectionsList(data.detections);

  document.getElementById("resultsSection").scrollIntoView({ behavior: "smooth" });
}

function drawDetectionsList(detections) {
  const listContainer = document.getElementById("detectionsList");
  const noDetectionsBox = document.getElementById("noDetections");

  if (detections.length === 0) {
    listContainer.hidden = true;
    noDetectionsBox.hidden = false;
    return;
  }

  listContainer.hidden = false;
  noDetectionsBox.hidden = true;
  listContainer.innerHTML = "";

  for (let i = 0; i < detections.length; i++) {
    const detection = detections[i];

    let barColorClass = "low";
    if (detection.confidence >= 90) {
      barColorClass = "high";
    } else if (detection.confidence >= 70) {
      barColorClass = "mid";
    }

    const row = document.createElement("div");
    row.className = "detection-item";
    row.innerHTML =
      '<span class="detection-class">' + detection.class + "</span>" +
      '<div class="detection-bar-track">' +
      '  <div class="detection-bar-fill ' + barColorClass + '" style="width: ' + detection.confidence + '%"></div>' +
      "</div>" +
      '<span class="detection-confidence">' + detection.confidence + "%</span>";

    listContainer.appendChild(row);
  }
}

function downloadPrediction() {
  if (!lastAnnotatedImage) {
    return;
  }
  const link = document.createElement("a");
  link.href = lastAnnotatedImage;
  link.download = "road-damage-prediction.jpg";
  link.click();
}

function newPrediction() {
  clearUpload(); 
  document.getElementById("resultsSection").hidden = true;
  document.getElementById("upload-section").scrollIntoView({ behavior: "smooth" });
}

document.addEventListener("DOMContentLoaded", function () {
  checkHealth();
  loadModels();
  setupUpload();

  document.getElementById("getStartedBtn").addEventListener("click", function () {
    document.getElementById("models-section").scrollIntoView({ behavior: "smooth" });
  });
  document.getElementById("downloadBtn").addEventListener("click", downloadPrediction);
  document.getElementById("newPredictionBtn").addEventListener("click", newPrediction);
});
