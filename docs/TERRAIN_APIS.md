# Terrain APIs — quick guide

These two endpoints turn a field boundary into map overlays of its terrain.
They show how the land slopes, where water flows and collects, and where soil is likely to erode.

Both endpoints need the header `x-api-key: <CROPGEN_API_KEY>`.

| API | Use |
|---|---|
| `POST /v4/api/terrain/{layer}` | Get **one** layer, e.g. only the slope map |
| `POST /v4/api/terrain` | Get **all 7** layers in one response |

## Request (same for both)

```json
{
  "aoi": { "type": "Polygon", "coordinates": [[[77.1566, 20.1358], [77.1575, 20.1356], [77.1573, 20.1348], [77.1564, 20.1348], [77.1566, 20.1358]]] },
  "start_date": "2026-08-01",
  "end_date": "2026-08-31"
}
```

- `aoi` is the field boundary as a GeoJSON Polygon or MultiPolygon (or a Feature containing one), in `[lon, lat]` order.
- The dates are used **only by `erosion_risk`**, which picks a satellite image from that range to measure crop cover. The other layers ignore them.

## The 7 layers

| Layer | What it shows |
|---|---|
| `elevation` | Height above sea level (m), with contour lines |
| `slope` | Steepness in %, from Flat to Steep |
| `aspect` | The direction the ground faces (N, NE, E…) |
| `water_flow` | Slope with arrows showing which way rain runs |
| `water_accumulation` | How much upslope area drains through each point; highlights drainage channels and hollows where water stands |
| `wetness` | Topographic Wetness Index (TWI): which parts stay dry and which collect water |
| `erosion_risk` | Low / Moderate / High soil-loss risk, from slope (RUSLE LS) and crop cover (satellite) |

## Response

A single-layer call returns:

```json
{
  "layer": "slope",
  "name": "Slope",
  "image_base64": "<PNG>",
  "bounds": { "west": 77.1563, "south": 20.1347, "east": 77.1575, "north": 20.1358 },
  "legend": { "type": "classes", "unit": "%", "items": [ { "label": "Flat", "range": "0–1 %", "color": "#1a9850" } ] },
  "warnings": ["..."]
}
```

- `image_base64` is a PNG that is transparent outside the field.
- `bounds` gives the image corners. Place the PNG exactly on them, e.g. with Leaflet `L.imageOverlay`.
- `legend` is what the frontend uses to draw the colour key.
- The all-layers call returns `{ bounds, layers: [ ...7 of the above... ], warnings }`.

## How it works

1. **Validate.** The API checks the polygon and dates: the field must be 0.05–500 ha and the dates can't be in the future.
2. **Get the elevation.** It downloads the Copernicus GLO-30 elevation model (30 m pixels) around the field.
3. **Compute the terrain.** It calculates slope, aspect, flow direction, flow accumulation and wetness.
4. **Cache.** Terrain results are stored for 24 h per field. If the frontend requests several layers at once, the terrain is computed only once and the other requests reuse it.
5. **Erosion only.** For `erosion_risk`, it also finds a satellite image in the date range to measure crop cover.
6. **Render.** It colours the layer, clips it to the field and returns a PNG with its legend.

**Tip:** Call single layers when the user switches tabs; after the first call, the rest come from cache.
Use the all-layers endpoint only when you need everything at once.

## Limits

- Elevation data is 30 m resolution and was surveyed in 2011–2015. A 1 ha field is only about 11 pixels, so detail inside it is smoothed (the `warnings` field says so).
- Fields levelled after 2015 will show the old terrain.
