# Face Recognition

A face recognition service: a **FastAPI** backend that enrols and identifies faces, and a
lightweight browser front end that streams webcam frames to it and draws the results.

> **Status:** enrolment and recognition work end to end over HTTP. The browser UI is a
> status dashboard for now; live webcam capture is next.

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
│ services/     encoder · gallery ·       │  ← plain Python, unit-testable
│               recognizer                │
├─────────────────────────────────────────┤
│ core/         config · logging · errors │
└─────────────────────────────────────────┘
```

**Why this split:** the services layer takes and returns arrays and dataclasses, so it is
tested without an HTTP client, and the detection backend can be swapped without touching the
routes. Blocking dlib calls are offloaded with `asyncio.to_thread`, so a slow frame never
stalls the event loop for other requests.

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
│   │   ├── uploads.py       # multipart -> decoded image
│   │   └── v1/
│   │       ├── router.py    # aggregates v1 routers
│   │       └── endpoints/   # health, faces, recognize
│   ├── schemas/             # common, face, recognition
│   ├── services/            # encoder, gallery, recognizer
│   └── utils/               # image decoding
├── frontend/                # static UI: index.html + css/ + js/
├── data/                    # gallery.db (gitignored)
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
| `MATCH_TOLERANCE` | `0.6` | Distance below which two faces are the same person. Lower is stricter. |
| `DETECTION_MODEL` | `hog` | `hog` (CPU, fast) or `cnn` (GPU, accurate). |
| `DETECTION_UPSAMPLE` | `1` | Higher finds smaller faces, costs time. |
| `GALLERY_DB_PATH` | `data/gallery.db` | SQLite file holding enrolled identities. |
| `MAX_UPLOAD_BYTES` | `5242880` | Rejects oversized uploads. |

## API

Versioned under `/api/v1`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Liveness — the process is up |
| `GET` | `/api/v1/health/ready` | Readiness — dependencies are usable |
| `POST` | `/api/v1/faces` | Enrol a face (multipart: `name`, `image`) |
| `GET` | `/api/v1/faces` | List enrolled identities |
| `DELETE` | `/api/v1/faces/{name}` | Remove an identity |
| `POST` | `/api/v1/recognize` | Identify faces in an image (multipart: `image`) |

### Health

```console
$ curl -s localhost:8000/api/v1/health/ready
{"ready":true,"checks":[
  {"name":"configuration","ready":true,"detail":null},
  {"name":"encoder","ready":true,"detail":"model loaded"},
  {"name":"gallery","ready":true,"detail":"2 identities enrolled"}]}
```

The two probes are deliberately distinct: **liveness** never touches dependencies, so a
restart-happy orchestrator cannot kill a healthy process over a transient outage.
**Readiness** answers `503` when any check fails, which takes the instance out of the load
balancer without restarting it. New dependencies are registered in `_run_checks()`.

### Enrolment

```console
$ curl -s -F name=nelay -F image=@me.jpg localhost:8000/api/v1/faces
{"name":"nelay","created_at":"2026-09-02T05:50:51.343490Z"}
```

Enrolment requires **exactly one face** in the image — zero returns `422 no_face_detected`,
several returns `422 multiple_faces`. A photo with two people is ambiguous about who is being
enrolled, so it is rejected rather than guessed at. Re-using a name returns `409`.

Responses never include the embedding: it is biometric data and no client needs it.

### Recognition

```console
$ curl -s -F image=@frame.jpg localhost:8000/api/v1/recognize
{"faces":[{"name":"nelay","known":true,"distance":0.3206,"confidence":0.4657,
           "box":{"top":1503,"right":1809,"bottom":2883,"left":429}}],
 "count":1,"processing_ms":124.4}
```

Returns **coordinates, not a rendered image** — the browser draws the overlay, so the server
never re-encodes a JPEG. Faces beyond `MATCH_TOLERANCE` come back as `"Unknown"` with
`distance: -1.0`, since JSON cannot represent infinity. An image with no faces is a
successful, empty result rather than an error.

`confidence` is a linear rescale of distance for display only — **not a calibrated
probability**. The tolerance comparison is what decides a match.

### Errors

Every failure returns the same envelope, so clients parse one shape:

```json
{ "code": "identity_exists", "message": "'nelay' is already enrolled." }
```

| Code | Status | Meaning |
| --- | --- | --- |
| `invalid_image` | 400 | Not decodable as an image |
| `identity_not_found` | 404 | No such enrolled name |
| `identity_exists` | 409 | Name already enrolled |
| `payload_too_large` | 413 | Over `MAX_UPLOAD_BYTES` or `MAX_IMAGE_PIXELS` |
| `no_face_detected` / `multiple_faces` | 422 | Enrolment needs exactly one face |

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
