# Face Recognition

A face recognition service: a **FastAPI** backend that enrols and identifies faces, and a
lightweight browser front end that streams webcam frames to it and draws the results.

> **Status:** working end to end — live webcam recognition in the browser, backed by a
> FastAPI enrolment and recognition API.

---

## Contents

- [Architecture](#architecture)
- [Project layout](#project-layout)
- [Getting started](#getting-started)
- [Accuracy](#accuracy)
- [Performance](#performance)
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
├── frontend/                # static UI
│   ├── index.html
│   ├── css/styles.css
│   └── js/                  # api, camera, overlay, app
├── data/                    # gallery.db (gitignored)
├── tests/                   # unit/ and integration/
├── scripts/                 # setup.sh (bootstrap), dev.sh (run), distance_check.py
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
./scripts/dev.sh
```

`dev.sh` reads `HOST`/`PORT` from `.env` (defaulting to `127.0.0.1:8013`) and passes them to
uvicorn. Running `uvicorn backend.app.main:app --reload` directly instead binds uvicorn's own
default of port 8000, ignoring `.env`.

The UI is served at <http://127.0.0.1:8013> and the interactive API docs at
<http://127.0.0.1:8013/docs>.

### Using it

1. **Enrol** — type a name, then either *Capture from camera* or *Upload a photo*.
   The photo must contain exactly one face.
2. **Start camera** — frames are sent to `/recognize` and boxes are drawn over the video.
   Green is a match, amber is `Unknown`.
3. **Remove** anyone from the *Enrolled* list with the × button.

Browsers only grant camera access over HTTPS or on `localhost`, so use `127.0.0.1` rather
than a LAN address when testing.

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

## Accuracy

### Reading the distance

Matching compares 128-dimension embeddings by Euclidean distance. **Lower is closer.** The
scale is not intuitive, so it is worth stating plainly:

| Distance | Meaning |
| --- | --- |
| `0.00` | Literally the same pixels — only happens when an image is matched against itself |
| `0.25`–`0.45` | **Same person**, different frame, lighting, angle or expression — the normal case |
| `~0.60` | dlib's published decision threshold, and this service's `MATCH_TOLERANCE` default |
| `0.70`–`1.00`+ | Different people |

Two separate photographs of the same person essentially never approach `0`: lighting, pose,
expression and JPEG noise all move the embedding. A frame captured seconds after enrolment
landing at `0.35` is a **confident match**, not a marginal one.

This matters because `confidence` is a linear rescale of distance, `1 - distance / tolerance`.
That maps the entire "confidently the same person" band onto roughly **0–50%**, and reserves
90%+ for distances under `0.06`, which no genuine second photograph reaches. So a real match
displaying as "40%" is expected and is *not* a sign of poor recognition. The `confidence`
field is for display only — **the tolerance comparison on `distance` is what decides a
match**, and `distance` is the number to trust when judging quality.

### Improving the match

Two settings trade time for tighter distances:

- **`ENCODING_MODEL`** (`large` by default) selects the landmark model used to align a face
  before describing it. `large` uses 68 points rather than `small`'s 5, so the crop handed to
  the descriptor is better centred and rotated. It must be identical for enrolment and
  recognition, or the two are not comparable.
- **`ENCODING_JITTERS` / `ENROLMENT_JITTERS`** re-describe a face several times with small
  random distortions and average the results, cancelling the noise of a single unlucky crop.
  Cost is linear in the count, hence a cheap `1` for live frames and a generous `10` for
  enrolment — enrolment runs once, and every later match is measured against the embedding it
  produces, so a noisy reference costs accuracy forever.

> **Changing `ENCODING_MODEL` invalidates an existing gallery.** Embeddings from the `small`
> and `large` models live in different spaces, so a probe described with one and a stored
> reference described with the other produce a meaningless distance — usually a wrong
> `Unknown`, with no error to signal it. After changing this setting, **re-enrol everyone**
> (delete each identity and add it again), or set `ENCODING_MODEL=small` to keep the old
> gallery working. Jitter counts are safe to change freely; they only average within the same
> space.

Beyond configuration, the largest remaining win is **enrolling each person more than once**,
from different angles and lighting. The gallery currently stores one embedding per name
(`name` is the table's primary key), so a single unrepresentative enrolment photo sets a
ceiling on every subsequent match.

No face photograph ships with this repository, so this cannot be measured in the test suite,
and no benchmark figures are quoted here. Measure it on your own images:

```bash
python scripts/distance_check.py me1.jpg me2.jpg          # same person
python scripts/distance_check.py me.jpg someone-else.jpg  # different people
```

It prints the distance and elapsed time under each configuration, old default first. A lower
number for the same person, and a higher one for two different people, is the improvement.

## Performance

Detection cost scales with pixel count, so a phone photo can take seconds while a webcam
frame takes milliseconds. Images are therefore shrunk so their longest edge is at most
`DETECTION_MAX_EDGE` (640px) before detection, and bounding boxes are scaled back to source
coordinates before they are returned. Measured on the same three images:

| Image | Uncapped | Capped at 640 |
| --- | --- | --- |
| 2592×4608 | 4548 ms | **189 ms** |
| 1650×1650 | 1355 ms | **258 ms** |
| 474×315 | 133 ms | 122 ms (below the cap, untouched) |

Accuracy is unaffected: the same-person distance stayed at 0.321 and different-person at
0.96, against a 0.6 tolerance.

Capping the *longest edge* is deliberate rather than scaling by a fixed ratio — a fixed ratio
either leaves large images slow or shrinks small ones until their faces disappear. Embeddings
are computed from the same downscaled frame during both enrolment and recognition, so
distances stay comparable.

The browser also caps captured frames at 640px and sends the next frame only once the
previous response arrives, so a slow server lowers the frame rate instead of queueing work.

### CPU and GPU

`COMPUTE_DEVICE` selects where detection runs:

| Value | Behaviour |
| --- | --- |
| `auto` *(default)* | Use the GPU if the installed dlib has CUDA and a device is visible; otherwise fall back to the CPU. |
| `gpu` | Require CUDA. **Refuses to start** if it is unavailable, rather than silently running an order of magnitude slower. |
| `cpu` | Force the CPU detector, without probing. |

The device selects the detector — `cnn` on GPU, `hog` on CPU — unless `DETECTION_MODEL` names
one explicitly, in which case that wins. `hog` is a classical CPU detector; `cnn` is a neural
one that is markedly better on rotated, small or poorly-lit faces but is impractically slow
without CUDA.

**A GPU only helps if dlib was built for it.** CUDA support is a property of the installed
wheel, not of the machine: the default `dlib-bin` wheel this project installs is CPU-only, so
a host with a perfectly good NVIDIA card still resolves to `cpu`. Using the GPU means
replacing that wheel with a CUDA-enabled dlib, which requires building from source against the
CUDA toolkit and cuDNN — there is no prebuilt CUDA wheel on PyPI.

#### Building a CUDA-enabled dlib (optional, not required to run this project)

This is a real compile (20–40 minutes) with a few prerequisites, so it is opt-in rather than
part of `setup.sh`. Skip it entirely unless you specifically want GPU-accelerated `cnn`
detection.

1. **NVIDIA driver** — already present if `nvidia-smi` shows your card. On WSL2 this comes
   from the *Windows* driver, not anything installed inside the Linux distro.
2. **CUDA Toolkit**, matching your driver (check `nvidia-smi`'s reported CUDA version). On
   WSL2:
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   rm cuda-keyring_1.1-1_all.deb   # installer artifact, not needed afterwards
   sudo apt-get update
   sudo apt-get install -y cuda-toolkit-12-8   # pick the series your driver supports
   ```
3. **cuDNN**, from the same NVIDIA apt repository the keyring above just registered:
   ```bash
   sudo apt-get install -y nvidia-cudnn
   ```
   Verify both landed before continuing — a missing cuDNN header fails the dlib build with an
   easy-to-miss CMake error, not an obvious "cuDNN not found":
   ```bash
   dpkg -l | grep -i cudnn
   find /usr -iname "libcudnn*.so*" -o -iname "cudnn*.h"
   ```
4. **Build dlib from source**, replacing the CPU-only wheel:
   ```bash
   pip uninstall -y dlib dlib-bin
   export PATH=/usr/local/cuda/bin:$PATH
   pip install dlib --no-binary dlib
   ```
5. **Confirm the build picked up CUDA:**
   ```bash
   python -c "import dlib; print(dlib.DLIB_USE_CUDA, dlib.cuda.get_num_devices())"
   ```
   `True` and a device count of at least `1` means `COMPUTE_DEVICE=auto` (or `gpu`) will now
   resolve to the GPU. If it still prints `False`, the CMake configure step didn't find CUDA or
   cuDNN — check its output for `USE_AVX_INSTRUCTIONS` / `DLIB_USE_CUDA` lines, which state the
   toolkit and cuDNN paths CMake decided on.

Check what actually resolved without reading the logs:

```console
$ curl -s localhost:8013/api/v1/health/ready | jq '.checks[] | select(.name=="encoder")'
{"name":"encoder","ready":true,
 "detail":"model loaded on cpu (hog) — falling back to CPU: the installed dlib wheel was built without CUDA support"}
```

Note that only *detection* moves to the GPU. The 128-d descriptor and the gallery comparison
run on the CPU either way, so on a webcam-sized frame — already capped at 640px, where `hog`
takes milliseconds — a GPU buys little. It pays off on large images, crowded frames, or when
`cnn`'s accuracy on awkward poses is what you are after.

## Configuration

All settings are environment variables, read once into a typed object in `core/config.py`.
See [`.env.example`](.env.example) for the full list. The ones worth tuning:

| Variable | Default | Purpose |
| --- | --- | --- |
| `ENVIRONMENT` | `development` | `development` \| `staging` \| `production`. Hides `/docs` in production. |
| `LOG_LEVEL` | `INFO` | Root logger level. |
| `CORS_ORIGINS` | localhost | Comma-separated browser origins. |
| `HOST` / `PORT` | `127.0.0.1:8013` | Bind address. |
| `MATCH_TOLERANCE` | `0.6` | Distance below which two faces are the same person. Lower is stricter. |
| `COMPUTE_DEVICE` | `auto` | `auto` (GPU if available, else CPU), `gpu` (require CUDA), `cpu` (force CPU). |
| `DETECTION_MODEL` | *(follows device)* | Explicit detector, overriding `COMPUTE_DEVICE`: `hog` (CPU) or `cnn` (GPU). |
| `ENCODING_MODEL` | `large` | Landmark model for alignment: `large` (68-point) or `small` (5-point). |
| `ENCODING_JITTERS` | `1` | Averaging passes per live frame. Higher is more accurate and slower. |
| `ENROLMENT_JITTERS` | `10` | Averaging passes when enrolling. Paid once per person. |
| `DETECTION_UPSAMPLE` | `1` | Higher finds smaller faces, costs time. |
| `DETECTION_MAX_EDGE` | `640` | Shrink to this before detecting; `0` disables. |
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
$ curl -s localhost:8013/api/v1/health/ready
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
$ curl -s -F name=nelay -F image=@me.jpg localhost:8013/api/v1/faces
{"name":"nelay","created_at":"2026-09-02T05:50:51.343490Z"}
```

Enrolment requires **exactly one face** in the image — zero returns `422 no_face_detected`,
several returns `422 multiple_faces`. A photo with two people is ambiguous about who is being
enrolled, so it is rejected rather than guessed at. Re-using a name returns `409`.

Responses never include the embedding: it is biometric data and no client needs it.

### Recognition

```console
$ curl -s -F image=@frame.jpg localhost:8013/api/v1/recognize
{"faces":[{"name":"nelay","known":true,"distance":0.3206,"confidence":0.4657,
           "box":{"top":1503,"right":1809,"bottom":2883,"left":429}}],
 "count":1,"processing_ms":124.4}
```

Returns **coordinates, not a rendered image** — the browser draws the overlay, so the server
never re-encodes a JPEG. Faces beyond `MATCH_TOLERANCE` come back as `"Unknown"` with
`distance: -1.0`, since JSON cannot represent infinity. An image with no faces is a
successful, empty result rather than an error.

`confidence` is a linear rescale of distance for display only — **not a calibrated
probability**. The tolerance comparison is what decides a match. A genuine match reads as
roughly 25–50%; see [Accuracy](#accuracy) for why, and read `distance` instead.

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
