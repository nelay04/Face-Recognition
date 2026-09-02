// getUserMedia lifecycle and frame capture.

/** Longest edge of a captured frame. The server shrinks to ~640 anyway, so
 *  sending more than this only costs upload time. */
const CAPTURE_MAX_EDGE = 640;

export class Camera {
  constructor(videoElement) {
    this.video = videoElement;
    this.stream = null;
    this.canvas = document.createElement("canvas");
  }

  get running() {
    return this.stream !== null;
  }

  /** Frame dimensions after capture scaling — the coordinate space the
   *  backend's bounding boxes come back in. */
  get captureSize() {
    const { videoWidth: w, videoHeight: h } = this.video;
    if (!w || !h) return { width: 0, height: 0 };

    const scale = Math.min(1, CAPTURE_MAX_EDGE / Math.max(w, h));
    return { width: Math.round(w * scale), height: Math.round(h * scale) };
  }

  async start() {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("This browser does not support camera access.");
    }

    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 } },
      audio: false,
    });

    this.video.srcObject = this.stream;
    await this.video.play();

    // play() can resolve before dimensions are known, and capture needs them.
    if (!this.video.videoWidth) {
      await new Promise((resolve) =>
        this.video.addEventListener("loadedmetadata", resolve, { once: true }),
      );
    }
  }

  stop() {
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.video.srcObject = null;
  }

  /** Grab the current frame as a JPEG blob, or null if not ready. */
  async capture(quality = 0.8) {
    const { width, height } = this.captureSize;
    if (!width || !height) return null;

    this.canvas.width = width;
    this.canvas.height = height;
    this.canvas.getContext("2d").drawImage(this.video, 0, 0, width, height);

    return new Promise((resolve) =>
      this.canvas.toBlob(resolve, "image/jpeg", quality),
    );
  }
}

/** Translate a common getUserMedia failure into something worth reading. */
export function describeCameraError(error) {
  switch (error?.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Camera permission was denied.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "No camera was found.";
    case "NotReadableError":
      return "The camera is already in use by another application.";
    default:
      return error?.message ?? "Could not start the camera.";
  }
}
