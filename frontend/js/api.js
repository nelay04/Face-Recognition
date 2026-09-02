// Thin wrapper around the backend REST API.

const BASE = "/api/v1";

/** Error carrying the backend's structured `code`, so callers can branch on it. */
export class ApiError extends Error {
  constructor(status, code, message) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${BASE}${path}`, options);
  } catch (cause) {
    throw new ApiError(0, "network_error", "Could not reach the server.");
  }

  if (response.status === 204) return null;

  const body = await response.json().catch(() => null);

  if (!response.ok) {
    // Our own errors carry {code, message}; FastAPI validation errors do not.
    throw new ApiError(
      response.status,
      body?.code ?? "invalid_request",
      body?.message ?? describeValidationError(body) ?? response.statusText,
    );
  }

  return body;
}

function describeValidationError(body) {
  const first = Array.isArray(body?.detail) ? body.detail[0] : null;
  if (!first) return null;
  const field = first.loc?.at(-1);
  return field ? `Invalid ${field}: ${first.msg}` : first.msg;
}

export const api = {
  health: () => request("/health"),
  readiness: () => request("/health/ready"),
  listFaces: () => request("/faces"),

  enrol(name, blob) {
    const form = new FormData();
    form.append("name", name);
    form.append("image", blob, "face.jpg");
    return request("/faces", { method: "POST", body: form });
  },

  deleteFace: (name) =>
    request(`/faces/${encodeURIComponent(name)}`, { method: "DELETE" }),

  recognize(blob) {
    const form = new FormData();
    form.append("image", blob, "frame.jpg");
    return request("/recognize", { method: "POST", body: form });
  },
};
