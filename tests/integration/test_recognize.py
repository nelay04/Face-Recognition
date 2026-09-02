"""Integration tests for the recognition endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.integration.conftest import BOX, StubFactory, detected

RECOGNIZE_URL = "/api/v1/recognize"
FACES_URL = "/api/v1/faces"


def recognize(client: TestClient, image: bytes) -> object:
    return client.post(
        RECOGNIZE_URL,
        files={"image": ("frame.png", image, "image/png")},
    )


def enrol(client: TestClient, name: str, image: bytes) -> None:
    response = client.post(
        FACES_URL,
        data={"name": name},
        files={"image": ("face.png", image, "image/png")},
    )
    assert response.status_code == 201


def test_recognises_an_enrolled_face(
    client: TestClient,
    stub_encoder: StubFactory,
    blank_png: bytes,
) -> None:
    stub_encoder([detected()])
    enrol(client, "nelay", blank_png)
    stub_encoder([detected(0.1)])

    response = recognize(client, blank_png)

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    [face] = body["faces"]
    assert face["name"] == "nelay"
    assert face["known"] is True
    assert 0.0 <= face["confidence"] <= 1.0


def test_returns_bounding_box_coordinates(
    client: TestClient,
    stub_encoder: StubFactory,
    blank_png: bytes,
) -> None:
    """The client draws the overlay, so coordinates come back, not an image."""
    stub_encoder([detected()])

    [face] = recognize(client, blank_png).json()["faces"]

    assert face["box"] == {
        "top": BOX.top,
        "right": BOX.right,
        "bottom": BOX.bottom,
        "left": BOX.left,
    }


def test_response_is_json_not_an_image(
    client: TestClient,
    stub_encoder: StubFactory,
    blank_png: bytes,
) -> None:
    stub_encoder([detected()])

    response = recognize(client, blank_png)

    assert response.headers["content-type"].startswith("application/json")


def test_unenrolled_face_is_unknown(
    client: TestClient,
    stub_encoder: StubFactory,
    blank_png: bytes,
) -> None:
    stub_encoder([detected()])

    [face] = recognize(client, blank_png).json()["faces"]

    assert face["name"] == "Unknown"
    assert face["known"] is False
    assert face["confidence"] == 0.0


def test_distant_face_is_unknown(
    client: TestClient,
    stub_encoder: StubFactory,
    blank_png: bytes,
) -> None:
    stub_encoder([detected()])
    enrol(client, "nelay", blank_png)
    stub_encoder([detected(5.0)])

    [face] = recognize(client, blank_png).json()["faces"]

    assert face["name"] == "Unknown"


def test_unknown_distance_is_json_safe(
    client: TestClient,
    stub_encoder: StubFactory,
    blank_png: bytes,
) -> None:
    """Infinity is not valid JSON, so it must be encoded as a sentinel."""
    stub_encoder([detected()])

    [face] = recognize(client, blank_png).json()["faces"]

    assert face["distance"] == -1.0


def test_image_with_no_faces_is_a_successful_empty_result(
    client: TestClient,
    stub_encoder: StubFactory,
    blank_png: bytes,
) -> None:
    """No face is a normal outcome for recognition, unlike for enrolment."""
    stub_encoder([])

    response = recognize(client, blank_png)

    assert response.status_code == 200
    assert response.json() == {
        "faces": [],
        "count": 0,
        "processing_ms": response.json()["processing_ms"],
    }


def test_handles_several_faces_in_one_frame(
    client: TestClient,
    stub_encoder: StubFactory,
    blank_png: bytes,
) -> None:
    stub_encoder([detected()])
    enrol(client, "nelay", blank_png)
    stub_encoder([detected(0.05), detected(9.0)])

    body = recognize(client, blank_png).json()

    assert body["count"] == 2
    assert [face["name"] for face in body["faces"]] == ["nelay", "Unknown"]


def test_reports_processing_time(
    client: TestClient,
    stub_encoder: StubFactory,
    blank_png: bytes,
) -> None:
    stub_encoder([detected()])

    body = recognize(client, blank_png).json()

    assert body["processing_ms"] >= 0.0


def test_undecodable_upload_returns_400(client: TestClient) -> None:
    response = recognize(client, b"definitely not an image")

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_image"


def test_missing_file_returns_422(client: TestClient) -> None:
    assert client.post(RECOGNIZE_URL).status_code == 422
