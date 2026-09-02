// Drawing recognition results over the video feed.

const KNOWN = "#12a150";
const UNKNOWN = "#d97706";

/**
 * Draw bounding boxes and labels.
 *
 * The canvas is sized to the *capture* dimensions, which is the coordinate
 * space the backend returns boxes in, and CSS stretches it over the video.
 * That keeps the drawing code free of scaling maths.
 */
export function drawFaces(canvas, faces, { mirrored = true } = {}) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.save();
  if (mirrored) {
    // The video is mirrored for a natural selfie view, so the overlay must be
    // mirrored too or the boxes land on the wrong side.
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
  }

  for (const face of faces) {
    drawFace(ctx, face, mirrored, canvas.width);
  }

  ctx.restore();
}

function drawFace(ctx, face, mirrored, canvasWidth) {
  const { top, right, bottom, left } = face.box;
  const colour = face.known ? KNOWN : UNKNOWN;
  const width = right - left;

  ctx.lineWidth = Math.max(2, Math.round(width / 60));
  ctx.strokeStyle = colour;
  ctx.strokeRect(left, top, width, bottom - top);

  const label = face.known
    ? `${face.name}  ${Math.round(face.confidence * 100)}%`
    : face.name;

  const fontSize = Math.max(13, Math.round(width / 9));
  ctx.font = `600 ${fontSize}px ui-sans-serif, system-ui, sans-serif`;

  const padding = fontSize * 0.4;
  const textWidth = ctx.measureText(label).width;
  const boxHeight = fontSize + padding * 2;
  // Keep the label on screen when the face is near the top edge.
  const labelTop = top - boxHeight < 0 ? bottom : top - boxHeight;

  ctx.fillStyle = colour;
  ctx.fillRect(left, labelTop, textWidth + padding * 2, boxHeight);

  // Flip the text back so it stays readable inside a mirrored canvas.
  ctx.save();
  if (mirrored) {
    ctx.translate(canvasWidth, 0);
    ctx.scale(-1, 1);
  }
  const textX = mirrored
    ? canvasWidth - (left + padding) - textWidth
    : left + padding;

  ctx.fillStyle = "#ffffff";
  ctx.textBaseline = "middle";
  ctx.fillText(label, textX, labelTop + boxHeight / 2);
  ctx.restore();
}

export function clearOverlay(canvas) {
  canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
}
