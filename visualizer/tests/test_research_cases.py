from __future__ import annotations

import base64
import csv
import gzip
import io
import struct
import zipfile
import zlib
from pathlib import Path
from xml.etree import ElementTree as ET

from fastapi.testclient import TestClient

from visualizer.backend.main import create_app
from visualizer.backend.replay_jobs import ReplayJobManager, _replay_id_for_path
from visualizer.backend.case_library import filter_cases

from test_replay_api import FakeCatalog, _bundle, _settings


def _research_root(tmp_path: Path) -> Path:
    root = tmp_path / "research"
    source = root / "uav_defend" / "envs" / "soldier_env.py"
    source.parent.mkdir(parents=True)
    source.write_text("# research-case API fixture\n", encoding="utf-8")
    return root


def _client(research_root: Path, output: Path, monkeypatch) -> tuple[TestClient, Path, ReplayJobManager]:
    import visualizer.backend.case_library as cases_module
    import visualizer.backend.main as main_module
    import visualizer.backend.replay_jobs as jobs_module
    import visualizer.backend.scientific_exports as exports_module

    output.mkdir(parents=True, exist_ok=True)
    for module in (cases_module, main_module, jobs_module, exports_module):
        monkeypatch.setattr(module, "local_output_root", lambda: output)
    replay_path = output / f"{'a' * 32}.json.gz"
    with gzip.open(replay_path, "wb") as stream:
        stream.write(_bundle().model_dump_json().encode("utf-8"))
    manager = ReplayJobManager(_settings(research_root), FakeCatalog())
    return TestClient(create_app(_settings(research_root), catalog_provider=FakeCatalog(), job_manager=manager)), replay_path, manager


def _png(width: int, height: int) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return len(payload).to_bytes(4, "big") + kind + payload + (zlib.crc32(kind + payload) & 0xffffffff).to_bytes(4, "big")

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    row = b"\x00" + bytes(width * 4)
    image_data = zlib.compress(row * height)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", image_data) + chunk(b"IEND", b"")


def test_case_api_bookmarks_exports_and_deletes_only_sidecar(tmp_path, monkeypatch):
    research_root = _research_root(tmp_path)
    client, replay_path, manager = _client(research_root, tmp_path / "local", monkeypatch)
    original_replay_bytes = replay_path.read_bytes()
    replay_id = _replay_id_for_path(replay_path)
    try:
        with client:
            created = client.post("/api/cases", json={
                "case_type": "single",
                "title": "Recorded seed case",
                "description": "API fixture",
                "tags": ["audit"],
                "replay_ids": [replay_id],
            })
            assert created.status_code == 201
            case = created.json()
            assert case["integrity"] == "valid"
            assert case["replay_references"][0]["artifact_sha256"]
            case_id = case["case_id"]

            bookmark = client.post(f"/api/cases/{case_id}/bookmarks", json={
                "expected_version": 1,
                "bookmark": {
                    "title": "Decision window",
                    "time_s": 0.25,
                    "frame_step": 0,
                    "decision_step": None,
                    "sync_mode": "absolute",
                    "local_time_a_s": 0.25,
                    "observation": "Recorded annotation",
                    "interpretation": "Researcher hypothesis only",
                    "tags": ["turn"],
                },
            })
            assert bookmark.status_code == 200
            updated = bookmark.json()
            assert updated["version"] == 2
            assert updated["bookmarks"][0]["label"] == "Decision window"

            stale = client.patch(f"/api/cases/{case_id}", json={"expected_version": 1, "notes": "stale"})
            assert stale.status_code == 409
            assert stale.json()["error"]["code"] == "CASE_VERSION_CONFLICT"

            exported = client.post(f"/api/cases/{case_id}/exports", json={
                "kind": "episode_csv",
                "start_time_s": 0,
                "end_time_s": 0.5,
            })
            assert exported.status_code == 200
            csv_file = next(item for item in exported.json()["files"] if item["name"] == "episode.csv")
            download = client.get(csv_file["download_url"])
            rows = list(csv.DictReader(io.StringIO(download.text)))
            assert [row["time_s"] for row in rows] == ["0.0", "0.5"]
            assert rows[-1]["requested_x"] == ""
            assert rows[-1]["decision_step"] == ""

            listing = client.get("/api/cases", params={"q": "audit", "architecture": "not-the-architecture"})
            assert listing.status_code == 200
            assert listing.json()["total"] == 0

            deleted = client.delete(f"/api/cases/{case_id}", params={"expected_version": 2})
            assert deleted.status_code == 204
            assert replay_path.read_bytes() == original_replay_bytes
            assert client.get(f"/api/cases/{case_id}").status_code == 404
    finally:
        manager.shutdown()


def test_case_integrity_reports_changed_replay_and_blocks_export(tmp_path, monkeypatch):
    research_root = _research_root(tmp_path)
    client, replay_path, manager = _client(research_root, tmp_path / "local", monkeypatch)
    replay_id = _replay_id_for_path(replay_path)
    try:
        with client:
            created = client.post("/api/cases", json={
                "case_type": "single", "title": "Integrity case", "replay_ids": [replay_id],
            }).json()
            replay_path.write_bytes(b"changed replay bytes")
            refreshed = client.get(f"/api/cases/{created['case_id']}").json()
            assert refreshed["integrity"] == "unresolved"
            export = client.post(f"/api/cases/{created['case_id']}/exports", json={"kind": "report"})
            assert export.status_code == 409
            assert export.json()["error"]["code"] == "REPLAY_HASH_MISMATCH"
    finally:
        manager.shutdown()


def test_scene_capture_saves_validated_dimensions_and_replay_provenance(tmp_path, monkeypatch):
    research_root = _research_root(tmp_path)
    client, replay_path, manager = _client(research_root, tmp_path / "local", monkeypatch)
    replay_id = _replay_id_for_path(replay_path)
    try:
        with client:
            case = client.post("/api/cases", json={
                "case_type": "single", "title": "Capture fixture", "replay_ids": [replay_id],
            }).json()
            png = _png(640, 360)
            response = client.post(f"/api/cases/{case['case_id']}/exports/scene-capture", json={
                "replay_id": replay_id,
                "replay_sha256": case["replay_references"][0]["artifact_sha256"],
                "png_base64": base64.b64encode(png).decode("ascii"),
                "width": 640,
                "height": 360,
                "time_s": 0.5,
                "camera": "free",
                "projection": "perspective",
            })
            assert response.status_code == 200
            capture = response.json()
            assert capture["width"] == 640
            assert capture["height"] == 360
            assert capture["replay_sha256"] == case["replay_references"][0]["artifact_sha256"]
            downloaded = client.get(capture["files"][0]["download_url"])
            assert downloaded.content == png
    finally:
        manager.shutdown()


def test_chart_capture_validates_svg_and_includes_dpi_png_in_package(tmp_path, monkeypatch):
    research_root = _research_root(tmp_path)
    client, replay_path, manager = _client(research_root, tmp_path / "local", monkeypatch)
    replay_id = _replay_id_for_path(replay_path)
    try:
        with client:
            case = client.post("/api/cases", json={
                "case_type": "single", "title": "Chart fixture", "replay_ids": [replay_id],
            }).json()
            case_id = case["case_id"]
            exported = client.post(f"/api/cases/{case_id}/exports", json={"kind": "trajectory_svg"}).json()
            svg_file = next(item for item in exported["files"] if item["name"].endswith(".svg"))
            svg_root = ET.fromstring(client.get(svg_file["download_url"]).content)
            view_box = [float(value) for value in svg_root.attrib["viewBox"].replace(",", " ").split()]
            png_height = round(2100 * view_box[3] / view_box[2])
            png = _png(2100, png_height)
            captured = client.post(
                f"/api/cases/{case_id}/exports/{exported['export_id']}/chart-capture",
                json={
                    "source_filename": svg_file["name"],
                    "source_sha256": svg_file["sha256"],
                    "png_base64": base64.b64encode(png).decode("ascii"),
                    "width": 2100,
                    "height": png_height,
                    "dpi": 300,
                },
            )
            assert captured.status_code == 200
            png_file = next(item for item in captured.json()["files"] if item["name"].endswith(".png"))
            png_bytes = client.get(png_file["download_url"]).content
            assert b"pHYs" in png_bytes
            phys_offset = png_bytes.index(b"pHYs")
            assert int.from_bytes(png_bytes[phys_offset + 4:phys_offset + 8], "big") == 11811
            assert png_bytes[phys_offset + 12] == 1

            package = client.post(f"/api/cases/{case_id}/exports", json={"kind": "package_zip"})
            assert package.status_code == 200
            zip_file = next(item for item in package.json()["files"] if item["name"].endswith(".zip"))
            with zipfile.ZipFile(io.BytesIO(client.get(zip_file["download_url"]).content)) as archive:
                assert f"charts_png/{png_file['name']}" in archive.namelist()
    finally:
        manager.shutdown()


def test_filter_cases_applies_independent_filters_without_architecture():
    cases = [{
        "case_type": "single",
        "title": "Case",
        "description": "",
        "notes": "",
        "tags": [],
        "bookmarks": [],
        "episodes": [{
            "defender": {"architecture": "analytic", "generation": None, "training_seed": None},
            "attacker": {"policy_id": "A0"},
            "environment": {"clutter": "C0", "awareness": "local"},
            "simulation_seed": 7,
            "outcome": "intercepted",
            "duration_s": 1.0,
        }],
    }]
    assert len(filter_cases(cases, clutter="C0")) == 1
    assert filter_cases(cases, clutter="C3") == []
    assert filter_cases(cases, attacker="A1-203") == []
