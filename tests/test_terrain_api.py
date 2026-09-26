"""
End-to-end API tests. No network: the DEM and satellite fetches are replaced
with a synthetic terrain, so everything after the download is exercised for real
(hydrology, rendering, reprojection, AOI clipping, PNG encoding, auth, validation).

    pytest -q tests/test_terrain_api.py
"""
import base64
import io
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

os.environ.setdefault("CROPGEN_API_KEY", "test-key-" + "x" * 32)

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

import main  # noqa: E402
import terrain_api as M  # noqa: E402
from terrain import engine as E  # noqa: E402

KEY = {"x-api-key": os.environ["CROPGEN_API_KEY"]}
URL = "/v4/api/terrain"

# the ~1 ha Washim plot
AOI = {"type": "Polygon", "coordinates": [[
    [77.15664790353888, 20.135857699751913], [77.15749548158759, 20.13559579948349],
    [77.15725944719428, 20.134800022897924], [77.15636895380133, 20.13477987660266],
    [77.15664790353888, 20.135857699751913]]]}
BODY = {"aoi": AOI, "start_date": "2026-06-01", "end_date": "2026-06-30"}
LAYERS = [l.value for l in M.Layer]


def _fake_dem(grid):
    H, W = grid["H"], grid["W"]
    yy, xx = np.mgrid[0:H, 0:W]
    rng = np.random.default_rng(0)
    dem = (515 + 0.012 * yy * grid["res_m"] + 0.006 * xx * grid["res_m"]
           - 2.5 * np.exp(-(((yy - H * .55) ** 2 + (xx - W * .55) ** 2) / (0.02 * H * W)))
           + rng.normal(0, 0.03, (H, W))).astype("float32")
    return dem, "synthetic", 30.0


CALLS = {"terrain": 0}


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    M._TERRAIN._data.clear()
    M._COVER._data.clear()
    CALLS["terrain"] = 0
    real = E.compute_terrain

    def counted(geom):
        CALLS["terrain"] += 1
        time.sleep(0.2)          # widen the race window for the single-flight test
        return real(geom)

    monkeypatch.setattr(E, "fetch_dem", _fake_dem)
    monkeypatch.setattr(E, "compute_terrain", counted)
    monkeypatch.setattr(E, "compute_cover",
                        lambda T, s, e: E.Cover(np.full(T.dem.shape, 0.2, "float32"),
                                                "test cover", "2026-06-15"))


client = TestClient(main.app)


def _png(b64):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA"))


# ── auth ─────────────────────────────────────────────────────────────────────
def test_missing_key_is_403():
    assert client.post(f"{URL}/slope", json=BODY).status_code == 403


def test_wrong_key_is_403():
    r = client.post(f"{URL}/slope", json=BODY, headers={"x-api-key": "nope"})
    assert r.status_code == 403


# ── validation ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("patch, fragment", [
    ({"start_date": "2026-07-01", "end_date": "2026-06-01"}, "start_date must be on or before"),
    ({"end_date": "2999-01-01"}, "future"),
    ({"start_date": "2010-01-01"}, "on or after"),
    ({"aoi": {"type": "Point", "coordinates": [77.1, 20.1]}}, "Polygon"),
    ({"aoi": {"type": "Polygon", "coordinates": [[[20.1, 95.0], [20.2, 95.0],
                                                  [20.2, 95.1], [20.1, 95.0]]]}}, "latitude"),
    ({"aoi": {"type": "Polygon", "coordinates": [[[77.1, 20.1], [77.101, 20.101],
                                                  [77.101, 20.1], [77.1, 20.101],
                                                  [77.1, 20.1]]]}}, "invalid"),
    ({"aoi": {"type": "Polygon", "coordinates": [[[77.1, 20.1], [77.10001, 20.1],
                                                  [77.10001, 20.10001], [77.1, 20.1]]]}},
     "minimum"),
])
def test_bad_requests_are_422(patch, fragment):
    r = client.post(f"{URL}/slope", json={**BODY, **patch}, headers=KEY)
    assert r.status_code == 422, r.text
    assert fragment in r.text


def test_unknown_layer_is_422():
    assert client.post(f"{URL}/ndvi", json=BODY, headers=KEY).status_code == 422


def test_feature_wrapper_accepted():
    body = {**BODY, "aoi": {"type": "Feature", "properties": {}, "geometry": AOI}}
    assert client.post(f"{URL}/slope", json=body, headers=KEY).status_code == 200


# ── output ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("layer", LAYERS)
def test_each_layer(layer):
    r = client.post(f"{URL}/{layer}", json=BODY, headers=KEY)
    assert r.status_code == 200, r.text
    d = r.json()
    assert set(d) >= {"layer", "name", "image_base64", "bounds", "legend"}
    assert d["layer"] == layer and d["name"]
    assert "legend" in d and d["legend"].get("type") in ("gradient", "classes")

    img = _png(d["image_base64"])
    h, w = img.shape[:2]
    assert max(h, w) == M.SETTINGS.output_px

    # bounds must be exactly the AOI's bounds
    xs = [p[0] for p in AOI["coordinates"][0]]
    ys = [p[1] for p in AOI["coordinates"][0]]
    b = d["bounds"]
    assert b["west"] == pytest.approx(min(xs)) and b["east"] == pytest.approx(max(xs))
    assert b["south"] == pytest.approx(min(ys)) and b["north"] == pytest.approx(max(ys))

    # clipped: the AOI is a skewed quad, so its bbox corners lie outside it
    assert img[0, -1, 3] == 0 and img[-1, -1, 3] == 0
    # and the centre is painted
    assert img[h // 2, w // 2, 3] == 255


def test_erosion_reports_its_source():
    d = client.post(f"{URL}/erosion_risk", json=BODY, headers=KEY).json()
    assert d["legend"]["source"] == "test cover"
    assert d["legend"]["scene_date"] == "2026-06-15"


def test_all_layers_endpoint():
    r = client.post(URL, json=BODY, headers=KEY)
    assert r.status_code == 200, r.text
    d = r.json()
    assert [l["layer"] for l in d["layers"]] == LAYERS
    assert CALLS["terrain"] == 1          # terrain computed once for all seven layers


def test_no_timing_fields_in_output():
    d = client.post(f"{URL}/slope", json=BODY, headers=KEY).json()
    blob = str(d).lower()
    assert "elapsed" not in blob and "duration" not in blob and "seconds" not in blob


# ── caching ──────────────────────────────────────────────────────────────────
def test_parallel_calls_compute_terrain_once():
    def hit(layer):
        return client.post(f"{URL}/{layer}", json=BODY, headers=KEY).status_code

    with ThreadPoolExecutor(max_workers=7) as ex:
        codes = list(ex.map(hit, LAYERS))
    assert codes == [200] * 7
    assert CALLS["terrain"] == 1


def test_different_geometry_same_session():
    other = {"type": "Polygon", "coordinates": [[
        [77.1345, 20.1180], [77.1392, 20.1178], [77.1394, 20.1142],
        [77.1341, 20.1145], [77.1345, 20.1180]]]}
    a = client.post(f"{URL}/elevation", json=BODY, headers=KEY)
    b = client.post(f"{URL}/elevation", json={**BODY, "aoi": other}, headers=KEY)
    assert a.status_code == b.status_code == 200
    assert a.json()["bounds"] != b.json()["bounds"]
    assert CALLS["terrain"] == 2
