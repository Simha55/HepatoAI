from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_bad_content_type():
    r = client.post("/detect", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415

def test_bad_image():
    r = client.post("/detect", files={"file": ("x.png", b"notanimage", "image/png")})
    assert r.status_code == 400
