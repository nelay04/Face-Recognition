// UI bootstrap: renders backend health status.

const API = "/api/v1";

const el = {
  dot: document.getElementById("dot"),
  statusText: document.getElementById("status-text"),
  meta: document.getElementById("meta"),
  app: document.getElementById("meta-app"),
  version: document.getElementById("meta-version"),
  env: document.getElementById("meta-env"),
  ready: document.getElementById("meta-ready"),
  refresh: document.getElementById("refresh"),
};

/**
 * GET a JSON endpoint. Readiness answers 503 when degraded, which is a
 * meaningful response rather than a failure, so the status is returned
 * alongside the body instead of being thrown away.
 */
async function getJSON(path) {
  const response = await fetch(`${API}${path}`, {
    headers: { Accept: "application/json" },
  });
  return { status: response.status, body: await response.json() };
}

function render(state, text) {
  el.dot.dataset.state = state;
  el.statusText.textContent = text;
}

async function refresh() {
  el.refresh.disabled = true;
  render("pending", "Checking…");

  try {
    const [health, ready] = await Promise.all([
      getJSON("/health"),
      getJSON("/health/ready"),
    ]);

    el.app.textContent = health.body.app;
    el.version.textContent = health.body.version;
    el.env.textContent = health.body.environment;
    el.ready.textContent = ready.body.ready ? "ready" : "not ready";
    el.meta.hidden = false;

    if (ready.body.ready) {
      render("ok", "Operational");
    } else {
      const failed = (ready.body.checks || [])
        .filter((check) => !check.ready)
        .map((check) => check.name)
        .join(", ");
      render("degraded", failed ? `Degraded — ${failed}` : "Degraded");
    }
  } catch {
    el.meta.hidden = true;
    render("error", "Unreachable");
  } finally {
    el.refresh.disabled = false;
  }
}

el.refresh.addEventListener("click", refresh);
refresh();
