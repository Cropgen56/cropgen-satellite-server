from fastapi import APIRouter, HTTPException
from models import CalculateRequest, CalculateResponse
import utils
import numpy as np
import time
from datetime import datetime, timedelta
import io
import base64

# rasterio + helpers (needed here because we call rasterio.open, Resampling, etc.)
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds, Window
from shapely.geometry import mapping

# concurrency helpers used in this module
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

router = APIRouter()

@router.post("/index", response_model=CalculateResponse)
def calculate_index(req: CalculateRequest):
    start_time = time.time()
    geom = req.geometry
    index_name = req.index_name.upper()
    date_str = req.date
    width = int(req.width or 800)
    height = int(req.height or 800)
    supersample = int(req.supersample or 4)  # Increased to 4 for smoother gradients
    smooth = bool(req.smooth if req.smooth is not None else True)  # Enable smoothing by default
    gaussian_sigma = float(req.gaussian_sigma or 3.0)  # Increased to 3.0 for stronger smoothing

    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")

    satellite = (req.satellite or "s2").lower()
    search_order = utils.get_provider_search_order(req.provider, prefer_pc_default=True)
    prefer_pc = search_order[0] == "planetary"

    # Sentinel-1 cannot compute optical indices like NDVI, EVI, etc.
    OPTICAL_INDICES = {
        "NDVI", "EVI", "EVI2", "SAVI", "MSAVI", "NDMI", "NDWI", "SMI",
        "CCC", "NITROGEN", "SOC", "NDRE", "RECI", "TRUE_COLOR"
    }
    if satellite.startswith("s1") and index_name in OPTICAL_INDICES:
        raise HTTPException(
            status_code=400,
            detail=f"Index {index_name} is not available for Sentinel-1 (radar). Use Sentinel-2 (s2) for optical indices."
        )

    index_band_map = {
        'NDVI': ['B04', 'B08'],
        'EVI': ['B04', 'B08', 'B02'],
        'EVI2': ['B04', 'B08'],
        'SAVI': ['B04', 'B08'],
        'MSAVI': ['B04', 'B08'],
        'NDMI': ['B08', 'B11'],
        'NDWI': ['B03', 'B11'],
        'SMI': ['B08', 'B11'],
        'CCC': ['B03', 'B04', 'B05'],
        'NITROGEN': ['B04', 'B05'],
        'SOC': ['B03', 'B04', 'B11', 'B12'],
        'NDRE': ['B8A', 'B05'],
        'RECI': ['B08', 'B05'],
        'TRUE_COLOR': ['B04', 'B03', 'B02'],
    }
    if index_name not in index_band_map:
        raise HTTPException(status_code=400, detail=f"Unsupported index: {index_name}")
    required_bands = index_band_map[index_name]

    try:
        item, has_scl, coll = utils.pick_best_item(geom, date_str, date_str, prefer_pc=prefer_pc, satellite=satellite)
        if not item:
            raise HTTPException(status_code=404, detail="No suitable item found for date/AOI")

        first_assets = item.assets or {}
        # pick a reasonable asset href
        first_candidate = utils.prefer_http_from_asset(first_assets.get("red") or first_assets.get("B04"))
        if not first_candidate:
            # try any asset
            for a in first_assets.values():
                h = utils.prefer_http_from_asset(a)
                if h:
                    first_candidate = h
                    break
        first_red = utils.sign_href_if_pc(first_candidate) if first_candidate else None
        if not first_red:
            raise HTTPException(status_code=500, detail="Could not determine reference band URL")
        with rasterio.open(first_red) as fr:
            target_crs = fr.crs

        aoi_sc, dst_transform, H, W, res_m = utils.build_adaptive_grid(target_crs, geom, native_res_m=10.0)

        coll_name = coll or utils.get_collections_for_satellite(satellite)[0]
        tiles = utils.items_for_date(geom, item.properties.get("datetime", ""), coll_name, prefer_pc=prefer_pc, limit=6) or [item]

        # precompute AOI mask (in scene CRS) for area stats & masking
        aoi_mask = geometry_mask([mapping(aoi_sc)], out_shape=(H, W), transform=dst_transform, invert=True)

        band_stack = {
            band: np.full((H, W), np.nan, dtype="float32")
            for band in required_bands
        }
        have_data = np.zeros((H, W), dtype=bool)
        S_stack = np.zeros((H, W), dtype="int16") if has_scl else None

        with ThreadPoolExecutor(max_workers=utils.THREADS) as ex:
            futures = [
                ex.submit(
                    utils._read_tile_into_stack,
                    it,
                    geom,
                    dst_transform,
                    H,
                    W,
                    S_stack is not None,
                    required_bands,
                )
                for it in tiles
            ]
            for fut in as_completed(futures):
                res = fut.result()
                if not res.get("used"):
                    continue
                bands = res["bands"]
                S = res.get("S")
                required_arrays = [bands.get(band) for band in required_bands]
                if any(arr is None for arr in required_arrays):
                    continue
                new_valid = np.logical_and.reduce(
                    [np.isfinite(arr) for arr in required_arrays]
                )
                take = new_valid & (~have_data)
                if np.any(take):
                    for band in required_bands:
                        band_stack[band][take] = bands[band][take]
                    if S_stack is not None and S is not None:
                        S_stack[take] = S[take]
                    have_data |= take

        band_dict = dict(band_stack)
        index_arr = None
        if index_name != "TRUE_COLOR":
            try:
                index_arr = utils.compute_index_array_by_name(index_name, band_dict)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Index compute error: {e}")

        # coverage check
        if index_name == "TRUE_COLOR":
            valid_pixels = np.count_nonzero(
                np.logical_and.reduce([np.isfinite(band_dict[band]) for band in required_bands]) & aoi_mask
            )
        else:
            valid_pixels = np.count_nonzero(np.isfinite(index_arr) & aoi_mask)
        valid_frac = valid_pixels / float(max(np.count_nonzero(aoi_mask), 1))

        if valid_frac < 0.7:
            fmt = "%Y-%m-%d"
            s = (datetime.strptime(date_str, fmt) - timedelta(days=14)).strftime(fmt)
            e = (datetime.strptime(date_str, fmt) + timedelta(days=14)).strftime(fmt)
            items_extra = []
            for provider_name in search_order:
                if provider_name == "planetary":
                    items_extra = utils.search_planetary([coll_name], geom, f"{s}/{e}", limit=12)
                else:
                    items_extra = utils.search_aws([coll_name], geom, f"{s}/{e}", limit=12)
                if items_extra:
                    break
            items_extra = [it for it in items_extra if getattr(it, "id", None) != getattr(item, "id", None)]
            try:
                median_bands = {}
                for band in required_bands:
                    median_bands[band] = utils.temporal_fill_median(
                        band,
                        items_extra,
                        geom,
                        dst_transform,
                        H,
                        W,
                        want_scl=False,
                        max_items=6,
                    )

                if all(median_bands[band] is not None for band in required_bands):
                    if index_name == "TRUE_COLOR":
                        fill_mask = (
                            ~np.logical_and.reduce(
                                [np.isfinite(band_dict[band]) for band in required_bands]
                            )
                        ) & aoi_mask
                    else:
                        fill_mask = (~np.isfinite(index_arr)) & aoi_mask

                    valid_fill = np.logical_and.reduce(
                        [np.isfinite(median_bands[band]) for band in required_bands]
                    )
                    fill_here = fill_mask & valid_fill
                    if np.any(fill_here):
                        for band in required_bands:
                            if np.isfinite(median_bands[band]).any() and np.nanmax(median_bands[band]) > 1.5:
                                median_bands[band] = median_bands[band] * (1 / 10000.0)
                            band_dict[band][fill_here] = median_bands[band][fill_here]

                        if index_name != "TRUE_COLOR":
                            index_arr = utils.compute_index_array_by_name(index_name, band_dict)
                            valid_pixels = np.count_nonzero(np.isfinite(index_arr) & aoi_mask)
                        else:
                            valid_pixels = np.count_nonzero(
                                np.logical_and.reduce(
                                    [np.isfinite(band_dict[band]) for band in required_bands]
                                ) & aoi_mask
                            )
                        valid_frac = valid_pixels / float(max(np.count_nonzero(aoi_mask), 1))
            except Exception:
                pass

        if S_stack is not None:
            classes = [8, 9, 10, 11]
            index_arr[np.isin(S_stack, classes)] = np.nan

        index_arr[~aoi_mask] = np.nan

        # ---- TRUE_COLOR special rendering ----
        if index_name == "TRUE_COLOR":
            R = band_dict.get("B04")
            G = band_dict.get("B03")
            B = band_dict.get("B02")
            if R is None or G is None or B is None:
                raise HTTPException(status_code=424, detail="True color requires B04,B03,B02 bands")

            valid_mask = np.isfinite(R) & np.isfinite(G) & np.isfinite(B) & aoi_mask
            if S_stack is not None:
                bad = np.isin(S_stack, [3, 8, 9, 10, 11])
                valid_mask = valid_mask & (~bad)

            if not np.any(valid_mask):
                raise HTTPException(status_code=424, detail="No valid pixels for TRUE_COLOR after masking")

            if np.nanmax(R) > 1.5 or np.nanmax(G) > 1.5 or np.nanmax(B) > 1.5:
                Rf = np.where(np.isfinite(R), R * (1/10000.0), 0.0)
                Gf = np.where(np.isfinite(G), G * (1/10000.0), 0.0)
                Bf = np.where(np.isfinite(B), B * (1/10000.0), 0.0)
            else:
                Rf = np.where(np.isfinite(R), R, 0.0)
                Gf = np.where(np.isfinite(G), G, 0.0)
                Bf = np.where(np.isfinite(B), B, 0.0)

            def stretch_channel(ch):
                vals = ch[valid_mask]
                if vals.size == 0: return ch
                lo = np.nanpercentile(vals, 2)
                hi = np.nanpercentile(vals, 98)
                if hi <= lo: return np.clip((ch - lo), 0.0, 1.0)
                out = (ch - lo) / (hi - lo)
                return np.clip(out, 0.0, 1.0)

            Rn, Gn, Bn = map(stretch_channel, [Rf, Gf, Bf])
            rgb = np.stack([Rn, Gn, Bn], axis=-1)
            rgb_uint8 = (rgb * 255.0).astype("uint8")

            # Create RGBA with transparency outside polygon
            rgba = np.zeros((H, W, 4), dtype=np.uint8)
            rgba[valid_mask, :3] = rgb_uint8[valid_mask]
            rgba[valid_mask, 3] = 255  # Full opacity inside polygon
            
            pil = Image.fromarray(rgba, mode="RGBA")
            # Use LANCZOS for high-quality resampling
            pil = pil.resize((width, height), resample=Image.LANCZOS)
            buf = io.BytesIO()
            pil.save(buf, format="PNG", optimize=True)
            img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            bounds = utils.compute_bounds_wgs84(dst_transform, W, H, target_crs)
            merged_legend = [{"color": "#000000", "label": "True Color", "hectares": 0.0, "percent": 0.0}]
            return {
                "date": date_str,
                "index_name": index_name,
                "image_base64": img_b64,
                "bounds": bounds,
                "legend": merged_legend
            }
        # ---- END TRUE_COLOR ----

        validvals = index_arr[np.isfinite(index_arr) & aoi_mask]
        if validvals.size == 0:
            raise HTTPException(status_code=424, detail="Index has no valid pixels after masking")

        bins_full = utils.ndvi_to_bins(index_arr)
        NDVI_canvas = index_arr

        if index_name in utils.index_palettes_labels:
            palette = utils.index_palettes_labels[index_name]['palette']
            labels = utils.index_palettes_labels[index_name]['labels']
        else:
            palette, labels = utils.PALETTE_MAP.get(index_name, (utils.PALETTE, utils.LABELS))

        # Pass AOI geometry, transform, and CRS for proper masking
        img_b64 = utils.render_spread_png_fast(
            bins_full, NDVI_canvas, res_m,
            supersample, smooth, gaussian_sigma,
            width, height, palette=palette, labels=labels,
            nodata_transparent=True,
            aoi_ll_geojson=geom,
            transform=dst_transform,
            crs=target_crs
        )

        legend = []
        for i in range(min(len(palette), len(labels))):
            legend.append({"color": palette[i], "label": labels[i]})
        for i in range(len(legend), len(palette)):
            legend.append({"color": palette[i], "label": f"bin_{i}"})

        bounds = utils.compute_bounds_wgs84(dst_transform, W, H, target_crs)

        pix_area_m2 = res_m * res_m
        area_stats = []
        aoi_pixel_count = int(np.count_nonzero(aoi_mask))
        max_bin = int(np.max(bins_full)) if bins_full.size > 0 else 0
        for bin_idx in range(0, max_bin + 1):
            if bin_idx == 0:
                label = "No Data"
            else:
                label = labels[bin_idx] if bin_idx < len(labels) else f"bin_{bin_idx}"
            cnt = int(np.count_nonzero((bins_full == bin_idx) & aoi_mask))
            ha = (cnt * pix_area_m2) / 10000.0
            pct = (cnt / float(max(aoi_pixel_count, 1))) * 100.0
            area_stats.append({"label": label, "hectares": round(float(ha), 4), "percent": round(float(pct), 2)})

        area_map = {a["label"]: a for a in area_stats}
        merged_legend = []
        for entry in legend:
            lbl = entry.get("label")
            am = area_map.get(lbl, {"hectares": 0.0, "percent": 0.0})
            merged_legend.append({
                "color": entry.get("color"),
                "label": lbl,
                "hectares": am.get("hectares", 0.0),
                "percent": am.get("percent", 0.0)
            })

        return {
            "date": date_str,
            "index_name": index_name,
            "image_base64": img_b64,
            "bounds": bounds,
            "legend": merged_legend
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))