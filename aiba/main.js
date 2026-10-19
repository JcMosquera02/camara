import * as faceapi from "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/dist/face-api.esm.js";

const MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/model";
const video = document.querySelector("#video");
const overlay = document.querySelector("#overlay");
const headGuide = document.querySelector("#head-guide");
const stage = document.querySelector("#camera-stage");
const placeholder = document.querySelector("#camera-placeholder");
const status = document.querySelector("#status");
const startButton = document.querySelector("#start-camera");
const stopButton = document.querySelector("#stop-camera");
const captureButton = document.querySelector("#capture-photo");
const captureCanvas = document.querySelector("#capture-canvas");

const STABILIZATION_SAMPLES = 8;
const MAX_MISSED_FRAMES = 10;
const HEAD_SILHOUETTE = [
  [50, 5], [44, 5], [37, 6], [31, 10], [27, 15], [24, 22], [22, 29], [21, 36],
  [22, 41], [23, 44], [21, 43], [19, 43], [18, 45], [18, 51], [20, 58], [21, 60],
  [25, 61], [27, 68], [31, 74], [38, 80], [38, 98], [62, 98], [62, 80], [69, 74],
  [73, 68], [75, 61], [79, 60], [80, 58], [82, 51], [82, 45], [81, 43], [79, 43],
  [77, 44], [78, 41], [79, 36], [78, 29], [76, 22], [73, 15], [69, 10], [63, 6], [56, 5],
].map(([x, y]) => [x * 0.78 + 11, y]);
let modelsPromise;
let startRequestId = 0;
let detectionFrameId;
let faceTracks = [];

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

function isFaceAligned(box) {
  const width = box.width / video.videoWidth;
  const height = box.height / video.videoHeight;
  const corners = [
    [box.x / video.videoWidth, box.y / video.videoHeight],
    [(box.x + box.width) / video.videoWidth, box.y / video.videoHeight],
    [box.x / video.videoWidth, (box.y + box.height) / video.videoHeight],
    [(box.x + box.width) / video.videoWidth, (box.y + box.height) / video.videoHeight],
  ];

  return width >= 0.18
    && width <= 0.5
    && height >= 0.25
    && height <= 0.65
    && corners.every((point) => isPointInsideHead(point));
}

function isPointInsideHead([x, y]) {
  const pointX = x * 100;
  const pointY = y * 100;
  let inside = false;

  for (let index = 0, previous = HEAD_SILHOUETTE.length - 1;
    index < HEAD_SILHOUETTE.length;
    previous = index, index += 1) {
    const [x1, y1] = HEAD_SILHOUETTE[index];
    const [x2, y2] = HEAD_SILHOUETTE[previous];
    const crossesY = (y1 > pointY) !== (y2 > pointY);
    const intersectionX = ((x2 - x1) * (pointY - y1)) / (y2 - y1) + x1;

    if (crossesY && pointX < intersectionX) {
      inside = !inside;
    }
  }

  return inside;
}

function traceHeadShape(context, width, height, closeAtNeck = false) {
  const scaleX = width / 100;
  const scaleY = height / 100;
  const guideX = (x) => (x * 0.78 + 11) * scaleX;
  const moveTo = (x, y) => context.moveTo(guideX(x), y * scaleY);
  const lineTo = (x, y) => context.lineTo(guideX(x), y * scaleY);
  const bezierCurveTo = (x1, y1, x2, y2, x3, y3) => {
    context.bezierCurveTo(
      guideX(x1),
      y1 * scaleY,
      guideX(x2),
      y2 * scaleY,
      guideX(x3),
      y3 * scaleY,
    );
  };

  context.beginPath();
  moveTo(50, 5);
  bezierCurveTo(34, 5, 25, 15, 23, 29);
  bezierCurveTo(21, 36, 22, 41, 23, 44);
  bezierCurveTo(19, 42, 17, 45, 18, 51);
  lineTo(20, 58);
  bezierCurveTo(21, 61, 22, 61, 25, 61);
  bezierCurveTo(27, 69, 31, 75, 38, 80);
  lineTo(38, 98);

  if (closeAtNeck) {
    lineTo(62, 98);
  } else {
    moveTo(62, 98);
  }

  lineTo(62, 80);
  bezierCurveTo(69, 75, 73, 69, 75, 61);
  bezierCurveTo(78, 61, 79, 61, 80, 58);
  lineTo(82, 51);
  bezierCurveTo(83, 45, 81, 42, 77, 44);
  bezierCurveTo(78, 41, 79, 36, 77, 29);
  bezierCurveTo(75, 15, 66, 5, 50, 5);

  if (closeAtNeck) {
    context.closePath();
  }
}

function maskOutsideHead(context, width, height) {
  context.save();
  context.fillStyle = "#000";
  context.beginPath();
  context.rect(0, 0, width, height);
  traceHeadShape(context, width, height, true);
  context.fill("evenodd");
  context.restore();
}

function trackDetections(detections) {
  const unmatchedTracks = new Set(faceTracks);
  const trackedDetections = detections.map((result) => {
    const box = result.detection.box;
    const centerX = box.x + box.width / 2;
    const centerY = box.y + box.height / 2;
    let track;
    let nearestDistance = Number.POSITIVE_INFINITY;

    for (const candidate of unmatchedTracks) {
      const candidateCenterX = candidate.box.x + candidate.box.width / 2;
      const candidateCenterY = candidate.box.y + candidate.box.height / 2;
      const distance = Math.hypot(centerX - candidateCenterX, centerY - candidateCenterY);
      const maxDistance = Math.max(
        80,
        Math.max(box.width, box.height, candidate.box.width, candidate.box.height) * 1.5,
      );

      if (distance < maxDistance && distance < nearestDistance) {
        track = candidate;
        nearestDistance = distance;
      }
    }

    if (track) {
      unmatchedTracks.delete(track);
    } else {
      track = { box, samples: 0, maleProbability: 0, age: 0, gender: null, missedFrames: 0 };
      faceTracks.push(track);
    }

    const maleProbability = result.gender === "male"
      ? result.genderProbability
      : 1 - result.genderProbability;
    if (track.samples < STABILIZATION_SAMPLES) {
      const sampleCount = track.samples + 1;
      track.maleProbability += (maleProbability - track.maleProbability) / sampleCount;
      track.age += (result.age - track.age) / sampleCount;
      track.samples = sampleCount;
      if (track.samples === STABILIZATION_SAMPLES) {
        track.gender = track.maleProbability >= 0.5 ? "Hombre" : "Mujer";
      }
    } else {
      const smoothing = 0.2;
      track.age += (result.age - track.age) * smoothing;
    }

    track.box = box;
    track.missedFrames = 0;

    return {
      box,
      gender: track.gender,
      age: Math.round(track.age),
      sampleCount: track.samples,
    };
  });

  for (const track of unmatchedTracks) {
    track.missedFrames += 1;
  }
  faceTracks = faceTracks.filter((track) => track.missedFrames <= MAX_MISSED_FRAMES);

  return trackedDetections;
}

function drawDetections(detections, message = "") {
  const context = overlay.getContext("2d");
  if (!context) {
    throw new Error("El navegador no pudo crear el lienzo para dibujar los rostros.");
  }

  context.clearRect(0, 0, overlay.width, overlay.height);

  const trackedDetections = trackDetections(detections);
  stage.classList.toggle("is-aligned", trackedDetections.length > 0);
  captureButton.disabled = trackedDetections.length !== 1;
  for (const face of trackedDetections) {
    const { x, y, width, height } = face.box;
    const fontSize = Math.max(14, Math.round(overlay.width / 50));
    const padding = 7;
    const label = face.gender
      ? `${face.gender} · edad ~${face.age} años`
      : "Ajustando rostro";
    const progress = face.sampleCount < STABILIZATION_SAMPLES
      ? `Ajustando estimación ${face.sampleCount}/${STABILIZATION_SAMPLES}`
      : "Estimación estabilizada";
    const labelHeight = fontSize * 2 + padding * 2;
    const labelY = Math.max(0, y - labelHeight);

    context.font = `600 ${fontSize}px system-ui, sans-serif`;
    const labelWidth = Math.max(
      context.measureText(label).width,
      context.measureText(progress).width,
    ) + padding * 2;
    context.fillStyle = "#31e6b0";
    context.fillRect(x, labelY, labelWidth, labelHeight);
    context.fillStyle = "#101827";
    context.fillText(label, x + padding, labelY + fontSize + padding / 2);
    context.font = `500 ${Math.max(11, fontSize - 2)}px system-ui, sans-serif`;
    context.fillText(progress, x + padding, labelY + fontSize * 2 + padding / 2);
  }

  if (detections.length) {
    const stillStabilizing = trackedDetections.some(
      (face) => face.sampleCount < STABILIZATION_SAMPLES,
    );
    status.textContent = stillStabilizing
      ? `Estabilizando edad y etiqueta con hasta ${STABILIZATION_SAMPLES} detecciones.`
      : `${detections.length} rostro${detections.length === 1 ? "" : "s"} con estimaciones estabilizadas.`;
  } else if (message) {
    status.textContent = message;
  } else {
    status.textContent = "Coloca una sola persona dentro del contorno para iniciar el escaneo.";
  }
}

async function detectFaces() {
  if (video.paused || video.ended || !video.videoWidth) {
    return;
  }

  try {
    const detections = await faceapi.detectAllFaces(
      video,
      new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.45 }),
    );
    const alignedFaces = detections.filter(({ detection }) => isFaceAligned(detection.box));

    if (alignedFaces.length === 1) {
      const faceCrops = await faceapi.extractFaces(
        video,
        alignedFaces.map(({ detection }) => detection),
      );
      const estimates = await Promise.all(
        faceCrops.map((faceCrop) => faceapi.predictAgeAndGender(faceCrop)),
      );
      drawDetections(alignedFaces.map((face, index) => ({
        ...face,
        ...estimates[index],
      })));
    } else if (alignedFaces.length > 1) {
      faceTracks = [];
      drawDetections([], "Hay más de una cara dentro del contorno. Deja solo una persona.");
    } else {
      faceTracks = [];
      drawDetections(
        [],
        detections.length
          ? "Rostro detectado fuera del contorno. Centra una sola persona en la guía."
          : "Coloca una sola persona dentro del contorno para iniciar el escaneo.",
      );
    }

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
  faceTracks = [];
  video.hidden = true;
  overlay.hidden = true;
  headGuide.hidden = true;
  placeholder.hidden = false;
  stage.classList.remove("is-aligned");
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
    captureButton.disabled = true;
    status.textContent = "Cargando detección facial…";

    await loadModels();
    if (requestId !== startRequestId) {
      return;
    }

    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    stage.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
    overlay.hidden = false;
    headGuide.hidden = false;
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
  if (!video.videoWidth || video.paused || captureButton.disabled) {
    status.textContent = "Centra una sola persona dentro del contorno antes de capturar.";
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
  maskOutsideHead(context, captureCanvas.width, captureCanvas.height);
  if (!overlay.hidden) {
    context.drawImage(overlay, 0, 0);
  }
  traceHeadShape(context, captureCanvas.width, captureCanvas.height);
  context.strokeStyle = "#fff";
  context.lineWidth = Math.max(2, captureCanvas.width / 400);
  context.lineCap = "round";
  context.lineJoin = "round";
  context.stroke();

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
