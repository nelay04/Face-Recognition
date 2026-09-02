// UI state and orchestration.

import { ApiError, api } from "./api.js";
import { Camera, describeCameraError } from "./camera.js";
import { clearOverlay, drawFaces } from "./overlay.js";

/** Floor between recognition requests. The loop is otherwise self-pacing:
 *  the next frame is only sent once the previous response has arrived, so a
 *  slow server naturally lowers the frame rate instead of queueing work. */
const MIN_FRAME_INTERVAL_MS = 120;

const el = (id) => document.getElementById(id);

const ui = {
  video: el("video"),
  overlay: el("overlay"),
  toggleCamera: el("toggle-camera"),
  cameraState: el("camera-state"),
  fps: el("fps"),
  latency: el("latency"),
  results: el("results"),
  name: el("enrol-name"),
  enrolCapture: el("enrol-capture"),
  enrolFile: el("enrol-file"),
  enrolPick: el("enrol-pick"),
  feedback: el("feedback"),
  gallery: el("gallery"),
  galleryCount: el("gallery-count"),
  statusDot: el("status-dot"),
  statusText: el("status-text"),
};

const camera = new Camera(ui.video);
let recognising = false;
let lastFrameAt = 0;

// ---------------------------------------------------------------- feedback

let feedbackTimer = null;

function notify(message, tone = "info") {
  ui.feedback.textContent = message;
  ui.feedback.dataset.tone = tone;
  ui.feedback.hidden = false;

  clearTimeout(feedbackTimer);
  feedbackTimer = setTimeout(() => {
    ui.feedback.hidden = true;
  }, 5000);
}

// ------------------------------------------------------------------ camera

async function toggleCamera() {
  ui.toggleCamera.disabled = true;
  try {
    if (camera.running) {
      stopRecognition();
    } else {
      await camera.start();
      sizeOverlay();
      startRecognition();
    }
    renderCameraState();
  } catch (error) {
    notify(describeCameraError(error), "error");
    renderCameraState();
  } finally {
    ui.toggleCamera.disabled = false;
  }
}

function sizeOverlay() {
  const { width, height } = camera.captureSize;
  ui.overlay.width = width;
  ui.overlay.height = height;
}

function renderCameraState() {
  const on = camera.running;
  ui.toggleCamera.textContent = on ? "Stop camera" : "Start camera";
  ui.toggleCamera.dataset.active = String(on);
  ui.cameraState.textContent = on ? "Live" : "Camera off";
  ui.enrolCapture.disabled = !on;
  document.body.dataset.cameraOn = String(on);

  if (!on) {
    ui.fps.textContent = "—";
    ui.latency.textContent = "—";
    ui.results.replaceChildren();
  }
}

// ------------------------------------------------------------- recognition

function startRecognition() {
  recognising = true;
  void recognitionLoop();
}

function stopRecognition() {
  recognising = false;
  camera.stop();
  clearOverlay(ui.overlay);
}

async function recognitionLoop() {
  while (recognising && camera.running) {
    const startedAt = performance.now();

    try {
      const blob = await camera.capture();
      if (!blob) break;

      const result = await api.recognize(blob);
      if (!recognising) break;

      drawFaces(ui.overlay, result.faces);
      renderResults(result.faces);
      renderTiming(startedAt, result.processing_ms);
    } catch (error) {
      // A single dropped frame is not worth tearing the session down;
      // only stop if the server has become unreachable.
      if (error instanceof ApiError && error.code === "network_error") {
        notify("Lost connection to the server.", "error");
        stopRecognition();
        renderCameraState();
        return;
      }
    }

    const elapsed = performance.now() - startedAt;
    if (elapsed < MIN_FRAME_INTERVAL_MS) {
      await new Promise((r) => setTimeout(r, MIN_FRAME_INTERVAL_MS - elapsed));
    }
  }
}

function renderTiming(startedAt, serverMs) {
  const now = performance.now();
  const roundTrip = now - startedAt;

  if (lastFrameAt) {
    const fps = 1000 / (now - lastFrameAt);
    ui.fps.textContent = `${fps.toFixed(1)} fps`;
  }
  lastFrameAt = now;
  ui.latency.textContent = `${Math.round(serverMs)} ms server · ${Math.round(roundTrip)} ms total`;
}

function renderResults(faces) {
  if (!faces.length) {
    ui.results.replaceChildren(chip("No faces detected", "muted"));
    return;
  }

  ui.results.replaceChildren(
    ...faces.map((face) =>
      chip(
        face.known
          ? `${face.name} · ${Math.round(face.confidence * 100)}%`
          : "Unknown",
        face.known ? "known" : "unknown",
      ),
    ),
  );
}

function chip(text, tone) {
  const span = document.createElement("span");
  span.className = "chip";
  span.dataset.tone = tone;
  span.textContent = text;
  return span;
}

// -------------------------------------------------------------- enrolment

async function enrolFrom(blobPromise) {
  const name = ui.name.value.trim();
  if (!name) {
    notify("Enter a name first.", "error");
    ui.name.focus();
    return;
  }

  try {
    const blob = await blobPromise;
    if (!blob) {
      notify("Could not capture an image.", "error");
      return;
    }

    const identity = await api.enrol(name, blob);
    notify(`Enrolled ${identity.name}.`, "success");
    ui.name.value = "";
    await loadGallery();
  } catch (error) {
    notify(enrolmentMessage(error), "error");
  }
}

/** Turn the backend's error codes into guidance the user can act on. */
function enrolmentMessage(error) {
  switch (error.code) {
    case "no_face_detected":
      return "No face found — move into frame and try again.";
    case "multiple_faces":
      return "More than one face in shot. Enrol one person at a time.";
    case "identity_exists":
      return "That name is already enrolled. Pick another, or delete it first.";
    case "invalid_image":
      return "That file could not be read as an image.";
    case "payload_too_large":
      return "That image is too large.";
    default:
      return error.message ?? "Enrolment failed.";
  }
}

// ---------------------------------------------------------------- gallery

async function loadGallery() {
  try {
    const { identities, count } = await api.listFaces();
    ui.galleryCount.textContent = String(count);

    if (!count) {
      const empty = document.createElement("p");
      empty.className = "empty";
      empty.textContent = "Nobody enrolled yet.";
      ui.gallery.replaceChildren(empty);
      return;
    }

    ui.gallery.replaceChildren(...identities.map(galleryRow));
  } catch {
    notify("Could not load the gallery.", "error");
  }
}

function galleryRow(identity) {
  const row = document.createElement("li");
  row.className = "identity";

  const name = document.createElement("span");
  name.className = "identity-name";
  name.textContent = identity.name;

  const date = document.createElement("time");
  date.className = "identity-date";
  date.dateTime = identity.created_at;
  date.textContent = new Date(identity.created_at).toLocaleDateString();

  const remove = document.createElement("button");
  remove.className = "icon-btn";
  remove.type = "button";
  remove.title = `Remove ${identity.name}`;
  remove.setAttribute("aria-label", `Remove ${identity.name}`);
  remove.textContent = "×";
  remove.addEventListener("click", () => deleteIdentity(identity.name));

  row.append(name, date, remove);
  return row;
}

async function deleteIdentity(name) {
  try {
    await api.deleteFace(name);
    notify(`Removed ${name}.`, "success");
    await loadGallery();
  } catch (error) {
    notify(error.message ?? "Could not remove that identity.", "error");
  }
}

// ----------------------------------------------------------------- status

async function refreshStatus() {
  try {
    const ready = await api.readiness();
    ui.statusDot.dataset.state = ready.ready ? "ok" : "degraded";
    ui.statusText.textContent = ready.ready ? "Service ready" : "Degraded";
  } catch {
    ui.statusDot.dataset.state = "error";
    ui.statusText.textContent = "Server unreachable";
  }
}

// ------------------------------------------------------------------- init

ui.toggleCamera.addEventListener("click", toggleCamera);
ui.enrolCapture.addEventListener("click", () => enrolFrom(camera.capture(0.92)));
ui.enrolPick.addEventListener("click", () => ui.enrolFile.click());
ui.enrolFile.addEventListener("change", () => {
  const [file] = ui.enrolFile.files;
  if (file) enrolFrom(Promise.resolve(file));
  ui.enrolFile.value = "";
});
ui.name.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && camera.running) ui.enrolCapture.click();
});

// Release the camera when the tab is hidden rather than holding the device.
document.addEventListener("visibilitychange", () => {
  if (document.hidden && camera.running) {
    stopRecognition();
    renderCameraState();
  }
});

renderCameraState();
void refreshStatus();
void loadGallery();
