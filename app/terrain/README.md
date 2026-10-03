# CropGen Terrain API

Elevation, slope, aspect, water flow, water accumulation, wetness and erosion
risk for any field, as AOI-clipped PNG overlays with legends.

Mounted in `main.py` under `/v4/api/terrain`, behind the server's shared
`x-api-key` (`CROPGEN_API_KEY`). Router: `app/routers/terrain_api.py`; engine: this package.

Tests (offline, no network): `pytest -q tests/test_terrain_api.py`

Optional env limits: `TERRAIN_MIN_AREA_HA` (0.05), `TERRAIN_MAX_AREA_HA` (500),
`TERRAIN_MAX_EXTENT_M` (5000), `TERRAIN_MAX_WINDOW_DAYS` (366),
`TERRAIN_CACHE_TTL_S` (86400), `TERRAIN_CACHE_MAX_ITEMS` (128), `TERRAIN_OUTPUT_PX` (768).

## Endpoints

| Method | Path | What |
|---|---|---|
| `POST` | `/v4/api/terrain/{layer}` | One layer |
| `POST` | `/v4/api/terrain` | All seven layers in one response |

`layer` is one of `elevation`, `slope`, `aspect`, `water_flow`,
`water_accumulation`, `wetness`, `erosion_risk`.

All calls need the header `x-api-key`.

### Request

```json
{
  "aoi": {"type": "Polygon", "coordinates": [[[77.1566479, 20.1358577], [77.1574955, 20.1355958],
          [77.1572594, 20.1348000], [77.1563690, 20.1347799], [77.1566479, 20.1358577]]]},
  "start_date": "2026-08-01",
  "end_date": "2026-08-31"
}
```

`aoi` may be a Polygon, a MultiPolygon, or a Feature wrapping either, in
`[lon, lat]` order. Only `erosion_risk` uses the dates: they pick the satellite
pass that supplies crop cover. Every other layer is pure terrain.

### Response (one layer)

```json
{
  "layer": "slope",
  "name": "Slope",
  "image_base64": "iVBORw0KGgo...",
  "bounds": {"west": 77.1563690, "south": 20.1347799, "east": 77.1574955, "north": 20.1358577},
  "legend": {
    "type": "classes",
    "unit": "%",
    "items": [
      {"label": "Flat", "range": "0–1 %", "color": "#1a9850"},
      {"label": "Very gentle", "range": "1–3 %", "color": "#91cf60"}
    ]
  },
  "warnings": ["Field covers about 11 native 30 m elevation pixels; ..."]
}
```

The PNG is in EPSG:4326 and its corners are exactly `bounds`; it is transparent
outside the AOI. `warnings` is omitted when there is nothing to say.

Legend shapes:

* `"type": "gradient"` — `min`, `max`, `unit`, `stops[{value, color}]`, optional `labels`
* `"type": "classes"` — `items[{label, range?, color}]`
* either may carry `symbols[{label, color, shape}]` for lines and markers drawn on top
* `erosion_risk` adds `source` and `scene_date` — which satellite pass drove it

### Put it on a map (Leaflet)

```js
const r = await fetch("/v4/api/terrain/slope", {method: "POST", headers: {...}, body});
const d = await r.json();
L.imageOverlay(`data:image/png;base64,${d.image_base64}`,
               [[d.bounds.south, d.bounds.west], [d.bounds.north, d.bounds.east]]).addTo(map);
```

**Call this API from your backend, not from browser JavaScript.** A key shipped
to the browser is a public key.

## Separate calls or one?

Separate. The first call for a field computes its terrain (DEM + hydrology) and
caches it; the other terrain layers then render from memory. Parallel calls for
the same field wait on the first rather than each computing it. Only
`erosion_risk` goes out to search satellite imagery.

`/v4/api/terrain` is there for when you genuinely want everything at once.

The cache is per process. With several uvicorn workers each keeps its own; move
it to Redis if that starts to matter.

## Thresholds to calibrate

These are fixed physical thresholds in `app/terrain/engine.py`, deliberately not
percentiles of each field — percentile classes painted "High erosion" on 18 % of
every field regardless of terrain. They are sensible starting points, not
calibrated values. Check them against fields your agronomists know.

| Constant | Default | Means |
|---|---|---|
| `EROSION_BREAKS` | `(0.5, 1.5)` | RUSLE LS × C: Low / Moderate / High |
| `DRAIN_AREA_M2` | `3000` | Upslope area where runoff forms a channel |
| `PONDING_MIN_M` | `0.15` | Hollow depth that counts as standing water |
| `FLAT_SLOPE_PCT` | `1.0` | Below this, aspect is shown as Flat |

## Limits

The elevation model is Copernicus GLO-30: 30 m pixels, surveyed 2011–2015, and a
surface model that includes tree canopy. A 1 ha field is about 11 of its pixels,
so detail inside a small field is interpolated and the API says so in
`warnings`. Fields levelled or bunded since 2015 will show the old terrain.
