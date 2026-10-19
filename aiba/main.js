import * as faceapi from "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/dist/face-api.esm.js";

const MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/model";
const video = document.querySelector("#video");
const overlay = document.querySelector("#overlay");
const stage = document.querySelector("#camera-stage");
const placeholder = document.querySelector("#camera-placeholder");
const status = document.querySelector("#status");
const startButton = document.querySelector("#start-camera");
const stopButton = document.querySelector("#stop-camera");
const captureButton = document.querySelector("#capture-photo");
const captureCanvas = document.querySelector("#capture-canvas");

let modelsPromise;
let startRequestId = 0;
let detectionFrameId;

function loadModels() {
  if (!modelsPromise) {
    modelsPromise = Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
      faceapi.nets.ageGenderNet.loadFromUri(MODEL_URL),
    ]).catch((error) => {
      modelsPromise = undefined;
      throw error;
    });
  }

  return modelsPromise;
}

function drawDetections(detections) {
  const context = overlay.getContext("2d");
  if (!context) {
    throw new Error("El navegador no pudo crear el lienzo para dibujar los rostros.");
  }

  context.clearRect(0, 0, overlay.width, overlay.height);

  for (const result of detections) {
    const { x, y, width, height } = result.detection.box;
    const label = result.gender === "male" ? "Hombre" : "Mujer";
    const confidence = Math.round(result.genderProbability * 100);
    const text = `Estimación: ${label} (${confidence}%)`;
    const fontSize = Math.max(14, Math.round(overlay.width / 50));
    const padding = 7;
    const labelHeight = fontSize + padding * 2;
    const labelY = Math.max(0, y - labelHeight);

    context.lineWidth = Math.max(2, Math.round(overlay.width / 320));
    context.strokeStyle = "#31e6b0";
    context.strokeRect(x, y, width, height);
    context.font = `600 ${fontSize}px system-ui, sans-serif`;
    const labelWidth = context.measureText(text).width + padding * 2;
    context.fillStyle = "#31e6b0";
    context.fillRect(x, labelY, labelWidth, labelHeight);
    context.fillStyle = "#101827";
    context.fillText(text, x + padding, labelY + fontSize + padding / 2);
  }

  status.textContent = detections.length
    ? `${detections.length} rostro${detections.length === 1 ? "" : "s"} detectado${detections.length === 1 ? "" : "s"}.`
    : "Buscando rostros…";
}

async function detectFaces() {
  if (video.paused || video.ended || !video.videoWidth) {
    return;
  }

  try {
    const detections = await faceapi
      .detectAllFaces(
        video,
        new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.45 }),
      )
      .withAgeAndGender();

    drawDetections(detections);
    detectionFrameId = window.requestAnimationFrame(detectFaces);
  } catch (error) {
    console.error("Error al analizar los rostros:", error);
    status.textContent = "No se pudo analizar la imagen de la cámara. Apágala e inténtalo de nuevo.";
  }
}

function stopCamera() {
  startRequestId += 1;
  if (detectionFrameId !== undefined) {
    window.cancelAnimationFrame(detectionFrameId);
    detectionFrameId = undefined;
  }

  const stream = video.srcObject;
  if (stream instanceof MediaStream) {
    stream.getTracks().forEach((track) => track.stop());
  }

  video.pause();
  video.srcObject = null;
  video.hidden = true;
  overlay.hidden = true;
  placeholder.hidden = false;
  overlay.getContext("2d")?.clearRect(0, 0, overlay.width, overlay.height);
  startButton.hidden = false;
  startButton.disabled = false;
  stopButton.hidden = true;
  captureButton.disabled = true;
  status.textContent = "La cámara está apagada.";
}

async function startCamera() {
  const requestId = ++startRequestId;
  startButton.disabled = true;
  status.textContent = "Solicitando acceso a la cámara…";

  if (!navigator.mediaDevices?.getUserMedia) {
    startButton.disabled = false;
    status.textContent = "Este navegador no permite acceder a la cámara. Abre la página en HTTPS o en localhost.";
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user" },
    });

    if (requestId !== startRequestId) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }

    video.srcObject = stream;
    await video.play();
    video.hidden = false;
    placeholder.hidden = true;
    startButton.hidden = true;
    stopButton.hidden = false;
    captureButton.disabled = false;
    status.textContent = "Cargando detección facial…";

    await loadModels();
    if (requestId !== startRequestId) {
      return;
    }

    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    stage.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
    overlay.hidden = false;
    status.textContent = "Buscando rostros…";
    detectFaces();
  } catch (error) {
    if (requestId !== startRequestId) {
      return;
    }

    console.error("No se pudo iniciar la cámara o la detección facial:", error);
    stopCamera();
    status.textContent = error.name === "NotAllowedError"
      ? "Permiso de cámara denegado. Autoriza el acceso desde el navegador e inténtalo de nuevo."
      : "No se pudo iniciar la cámara o cargar la detección facial. Revisa la conexión, los permisos y vuelve a intentarlo.";
  }
}

function capturePhoto() {
  if (!video.videoWidth || video.paused) {
    status.textContent = "Enciende la cámara antes de capturar una foto.";
    return;
  }

  captureCanvas.width = video.videoWidth;
  captureCanvas.height = video.videoHeight;
  const context = captureCanvas.getContext("2d");
  if (!context) {
    status.textContent = "No se pudo preparar la captura.";
    return;
  }

  context.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
  if (!overlay.hidden) {
    context.drawImage(overlay, 0, 0);
  }

  captureCanvas.toBlob((blob) => {
    if (!blob) {
      status.textContent = "No se pudo generar la foto.";
      return;
    }

    const downloadUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = "captura-camara.jpg";
    link.click();
    URL.revokeObjectURL(downloadUrl);
    status.textContent = "Foto capturada y descargada.";
  }, "image/jpeg", 0.92);
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
captureButton.addEventListener("click", capturePhoto);
