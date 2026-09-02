// Drawing recognition results over the video feed.

// Vivid enough to stay legible over arbitrary camera footage — the muted UI
// palette washes out against a bright video frame.
const TONES = {
  known: { stroke: "#1479ff", label: "#1479ff", text: "#ffffff" },
  unknown: { stroke: "#ff9500", label: "#ff9500", text: "#1c1c1a" },
};

const LABEL_GAP = 6;
const MIN_FONT = 12;
const MAX_FONT = 17;

/**
 * Match the canvas backing store to the density it is actually painted at.
 *
 * The canvas keeps the *capture* aspect ratio so that `object-fit: cover`
 * crops it exactly the way it crops the video underneath. Only the pixel
 * density changes: a 640px-wide capture stretched across a 900px CSS box
 * used to be upscaled by the compositor, which is what made the strokes
 * and text look soft.
 *
 * Returns the density, which `drawFaces` reads back off the element.
 */
export function resizeOverlay(canvas, capture) {
  if (!capture.width || !capture.height) return 0;

  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const cover = rect.width
    ? Math.max(rect.width / capture.width, rect.height / capture.height)
    : 1;
  // Capped: past ~3x the extra pixels cost fill rate without being visible.
  const density = Math.min(Math.max(cover * dpr, 1), 3);

  canvas.width = Math.round(capture.width * density);
  canvas.height = Math.round(capture.height * density);
  canvas.dataset.density = String(density);
  return density;
}

/**
 * Draw bounding boxes and labels.
 *
 * Drawing happens in capture coordinates — the space the backend returns
 * boxes in — with the context scaled up to the backing store's density.
 */
export function drawFaces(canvas, faces, { mirrored = true } = {}) {
  const ctx = canvas.getContext("2d");
  const density = Number(canvas.dataset.density) || 1;

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.scale(density, density);

  const width = canvas.width / density;
  for (const face of faces) {
    drawFace(ctx, face, width, mirrored);
  }
}

function drawFace(ctx, face, canvasWidth, mirrored) {
  const { top, right, bottom, left } = face.box;
  const tone = face.known ? TONES.known : TONES.unknown;

  // The video is mirrored for a natural selfie view, so the box has to be
  // too. Reflecting the *coordinates* rather than the canvas keeps the label
  // text the right way round — flipping the canvas reverses the glyphs.
  const x = mirrored ? canvasWidth - right : left;
  const w = right - left;
  const h = bottom - top;

  ctx.lineWidth = Math.min(Math.max(w / 110, 1.5), 3);
  ctx.strokeStyle = tone.stroke;
  // Half-pixel offset so a whole-pixel stroke lands on one row of pixels
  // instead of straddling two and rendering as a soft double line.
  const inset = ctx.lineWidth / 2;
  ctx.strokeRect(x + inset, top + inset, w - ctx.lineWidth, h - ctx.lineWidth);

  drawLabel(ctx, face, tone, x, top, bottom, w);
}

function drawLabel(ctx, face, tone, x, top, bottom, w) {
  const label = face.known
    ? `${face.name} · ${Math.round(face.confidence * 100)}%`
    : face.name;

  // Tied loosely to the box so it tracks distance, but clamped — scaling
  // straight off the width turned a close-up face into a wall of text.
  const fontSize = Math.min(Math.max(Math.round(w / 14), MIN_FONT), MAX_FONT);
  ctx.font = `500 ${fontSize}px ui-sans-serif, system-ui, sans-serif`;
  ctx.textBaseline = "middle";

  const padX = Math.round(fontSize * 0.55);
  const textWidth = ctx.measureText(label).width;
  const plateW = textWidth + padX * 2;
  const plateH = Math.round(fontSize * 1.75);
  // Sit above the box, or below it when the face is near the top edge.
  const plateY =
    top - LABEL_GAP - plateH < 0 ? bottom + LABEL_GAP : top - LABEL_GAP - plateH;

  ctx.fillStyle = tone.label;
  ctx.beginPath();
  ctx.roundRect(x, plateY, plateW, plateH, 4);
  ctx.fill();

  ctx.fillStyle = tone.text;
  ctx.fillText(label, x + padX, plateY + plateH / 2);
}

export function clearOverlay(canvas) {
  const ctx = canvas.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}
