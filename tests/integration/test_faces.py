"""Integration tests for the enrolment endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.integration.conftest import StubFactory, detected

FACES_URL = "/api/v1/faces"


def upload(client: TestClient, name: str, image: bytes) -> object:
    return client.post(
        FACES_URL,
        data={"name": name},
        files={"image": ("face.png", image, "image/png")},
    )


def test_enrol_returns_201_with_the_identity(
    client: TestClient,
    stub_encoder: StubFactory,
    blank_png: bytes,
) -> None:
    stub_encoder([detected()])

    response = upload(client, "nelay", blank_png)

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "nelay"
    assert "created_at" in body


def test_enrolment_never_exposes_the_embedding(
    client: TestClient,
    stub_encoder: StubFactory,
    blank_png: bytes,
) -> None:
    """Embeddings are biometric data and must not leave the server."""
    stub_encoder([detected()])

    body = upload(client, "nelay", blank_png).json()

    assert "embedding" not in body


def test_duplicate_name_returns_409(
    client: TestClient,
    stub_encoder: StubFactory,
    blank_png: bytes,
) -> None:
    stub_encoder([detected()])
    upload(client, "nelay", blank_png)

    response = upload(client, "nelay", blank_png)

    assert response.status_code == 409
    assert response.json()["code"] == "identity_exists"


def test_image_with_no_face_returns_422(
    client: TestClient,
    stub_encoder: StubFactory,
    blank_png: bytes,
) -> None:
    stub_encoder([])

    response = upload(client, "nelay", blank_png)

    assert response.status_code == 422
    assert response.json()["code"] == "no_face_detected"


def test_image_with_several_faces_returns_422(
    client: TestClient,
    stub_encoder: StubFactory,
    blank_png: bytes,
) -> None:
    """Enrolment must refuse to guess which person to store."""
    stub_encoder([detected(), detected(1.0)])

    response = upload(client, "nelay", blank_png)

    assert response.status_code == 422
    assert response.json()["code"] == "multiple_faces"


def test_undecodable_upload_returns_400(client: TestClient) -> None:
    response = upload(client, "nelay", b"definitely not an image")

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_image"


def test_invalid_name_is_rejected_before_any_work(
    client: TestClient,
    blank_png: bytes,
) -> None:
    response = upload(client, "bad/name", blank_png)

    assert response.status_code == 422


def test_list_is_empty_initially(client: TestClient) -> None:
    response = client.get(FACES_URL)

    assert response.status_code == 200
    assert response.json() == {"identities": [], "count": 0}


def test_list_returns_enrolled_identities(
    client: TestClient,
    stub_encoder: StubFactory,
    blank_png: bytes,
) -> None:
    stub_encoder([detected()])
    upload(client, "nelay", blank_png)
    stub_encoder([detected(1.0)])
    upload(client, "messi", blank_png)

    body = client.get(FACES_URL).json()

    assert body["count"] == 2
    assert {item["name"] for item in body["identities"]} == {"nelay", "messi"}


def test_delete_removes_the_identity(
    client: TestClient,
    stub_encoder: StubFactory,
    blank_png: bytes,
) -> None:
    stub_encoder([detected()])
    upload(client, "nelay", blank_png)

    response = client.delete(f"{FACES_URL}/nelay")

    assert response.status_code == 204
    assert client.get(FACES_URL).json()["count"] == 0


def test_delete_unknown_identity_returns_404(client: TestClient) -> None:
    response = client.delete(f"{FACES_URL}/nobody")

    assert response.status_code == 404
    assert response.json()["code"] == "identity_not_found"


def test_enrolment_survives_a_restart(
    client: TestClient,
    stub_encoder: StubFactory,
    blank_png: bytes,
) -> None:
    """The gallery is on disk, so identities outlive the process."""
    stub_encoder([detected()])
    upload(client, "nelay", blank_png)

    # The client fixture keeps one settings object, so a fresh app points at
    # the same database file.
    from backend.app.main import create_app

    with TestClient(create_app(client.app.state.settings)) as restarted:
        assert restarted.get(FACES_URL).json()["count"] == 1
