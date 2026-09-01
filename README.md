# Face Recognition

A face recognition service: a **FastAPI** backend that enrols and identifies faces, and a
lightweight browser front end that streams webcam frames to it and draws the results.

> **Status:** the health API is implemented, tested and running. Enrolment and recognition
> are next; the structure and dependency set are already in place for them.

---

## Contents

- [Architecture](#architecture)
- [Project layout](#project-layout)
- [Getting started](#getting-started)
- [Configuration](#configuration)
- [API](#api)
- [Development](#development)
- [Privacy](#privacy)

---

## Architecture

The backend is layered so that the recognition logic never depends on the web framework:

The backend is layered so that domain logic never depends on the web framework:

```
Browser (static UI)
        │  JSON over HTTP
        ▼
┌─────────────────────────────────────────┐
│ api/          routing, validation, HTTP │  ← FastAPI lives only here
├─────────────────────────────────────────┤
│ schemas/      request & response models │
├─────────────────────────────────────────┤
│ services/     domain logic  (planned)   │  ← plain Python, unit-testable
├─────────────────────────────────────────┤
│ core/         config · logging · errors │
└─────────────────────────────────────────┘
```

**Why this split:** the eventual services layer will take and return arrays and dataclasses,
so it can be tested without an HTTP client, and the detection backend can be swapped without
touching the routes.

Two conventions are worth knowing:

- **`create_app()` is a factory**, not a module-level singleton. Tests build an isolated app
  with their own `Settings` rather than mutating global state.
- **Errors go through `AppError`**, so every failure renders the same `ErrorResponse` body and
  unexpected exceptions are logged server-side without leaking internals to the client.

## Project layout

```
.
├── backend/app/
│   ├── main.py              # create_app() factory + ASGI entrypoint
│   ├── core/                # config, logging, exceptions
│   ├── api/
│   │   ├── deps.py          # injectable dependencies
│   │   └── v1/
│   │       ├── router.py    # aggregates v1 routers
│   │       └── endpoints/   # health.py
│   └── schemas/             # common.py (health + error envelopes)
├── frontend/                # static UI: index.html + css/ + js/
├── tests/                   # unit/ and integration/
├── scripts/                 # setup.sh (bootstrap), dev.sh (run)
└── requirements/            # base.txt, dev.txt
```

## Getting started

**Requirements:** Python 3.12+ and a C++ runtime (prebuilt dlib wheels are used, so no
compiler is needed).

```bash
git clone git@github.com:nelay04/Face-Recognition.git
cd Face-Recognition

./scripts/setup.sh --dev
source .venv/bin/activate

cp .env.example .env
```

Run it:

```bash
uvicorn backend.app.main:app --reload
```

The UI is served at <http://127.0.0.1:8000> and the interactive API docs at
<http://127.0.0.1:8000/docs>.

### About the dependency install

`face_recognition` and its model package are unmaintained and have two metadata problems that
`scripts/setup.sh` works around. If you install by hand, the order matters:

| Problem | Effect | Fix |
| --- | --- | --- |
| `face-recognition` requires `dlib>=19.7` | pip fetches the **source-only** `dlib` and starts a long CMake build that shadows the prebuilt wheel | depend on `dlib-bin`; install `face-recognition` with `--no-deps` |
| `face_recognition_models` imports `pkg_resources` | `pkg_resources` is absent from 3.12+ venvs and gone from setuptools 81; `import face_recognition` then prints a *misleading* "install face_recognition_models" error | pin `setuptools<81` |

So the manual equivalent is:

```bash
pip install -r requirements/base.txt
pip install --no-deps face-recognition==1.3.0
```

## Configuration

All settings are environment variables, read once into a typed object in `core/config.py`.
See [`.env.example`](.env.example) for the full list. The ones worth tuning:

| Variable | Default | Purpose |
| --- | --- | --- |
| `ENVIRONMENT` | `development` | `development` \| `staging` \| `production`. Hides `/docs` in production. |
| `LOG_LEVEL` | `INFO` | Root logger level. |
| `CORS_ORIGINS` | localhost | Comma-separated browser origins. |
| `HOST` / `PORT` | `127.0.0.1:8000` | Bind address. |

Recognition tuning (`MATCH_TOLERANCE`, `DETECTION_MODEL`, storage paths) arrives with the
recognition endpoints.

## API

Versioned under `/api/v1`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Liveness — the process is up |
| `GET` | `/api/v1/health/ready` | Readiness — dependencies are usable |

```console
$ curl -s localhost:8000/api/v1/health
{"status":"ok","app":"Face Recognition API","version":"0.1.0","environment":"development"}

$ curl -s localhost:8000/api/v1/health/ready
{"ready":true,"checks":[{"name":"configuration","ready":true,"detail":null}]}
```

The two probes are deliberately distinct: **liveness** never touches dependencies, so a
restart-happy orchestrator cannot kill a healthy process over a transient outage.
**Readiness** answers `503` when any check fails, which takes the instance out of the load
balancer without restarting it. New dependencies are registered in `_run_checks()`.

Enrolment (`/faces`) and recognition (`/recognize`) are planned.

## Development

```bash
ruff check . && ruff format --check .   # lint & formatting
mypy                                    # type checking
pytest                                  # tests with coverage
```

Install the pre-commit hooks so the above runs before each commit:

```bash
pre-commit install
```

## Privacy

Face embeddings and enrolment photos are **biometric personal data**.

- `.gitignore` excludes image formats tree-wide, so a stray photo cannot be committed by
  accident, along with `.env` and the virtualenv.
- No training or sample faces ship with this repository.
- When enrolment lands: enrol only people who have consented, and support deletion on request.

## License

MIT — see [LICENSE](LICENSE).
