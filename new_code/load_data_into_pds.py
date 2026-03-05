from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from pynwb import NWBHDF5IO
from bids import BIDSLayout
import nibabel as nib


# -----------------------
# subject mapping (int -> ids)
# -----------------------
def bids_subject_from_int(sub: int) -> str:
    return f"p{sub}cs"   # BIDS folder: sub-p41cs


def nwb_subject_from_int(sub: int) -> str:
    return f"CS{sub}"    # NWB folder/file: sub-CS41


# -----------------------
# NWB helpers
# -----------------------
def find_nwb_file_for_subject(nwb_root: Union[str, Path], nwb_sub: str) -> Path:
    nwb_root = Path(nwb_root)
    if not nwb_root.exists():
        raise FileNotFoundError(f"NWB root not found: {nwb_root}")

    patterns = [
        f"**/sub-{nwb_sub}/*.nwb",
        f"**/sub-{nwb_sub}*.nwb",
        f"**/*sub-{nwb_sub}*.nwb",
    ]
    hits: List[Path] = []
    for pat in patterns:
        hits.extend(nwb_root.glob(pat))
    hits = sorted(set(hits))

    if not hits:
        all_nwb = list(nwb_root.rglob("*.nwb"))
        hits = sorted([p for p in all_nwb if f"sub-{nwb_sub}" in str(p)])

    if not hits:
        raise FileNotFoundError(f"No NWB found for subject {nwb_sub} under {nwb_root}")

    return hits[0]


def _read_timeseries(ts, max_timepoints: Optional[int] = None) -> np.ndarray:
    """Read NWB TimeSeries-like object into np array (optionally truncate first dimension)."""
    if max_timepoints is None:
        return np.asarray(ts.data[:])
    return np.asarray(ts.data[:max_timepoints])


def load_nwb_lfp(nwbfile, which: str = "LFP_macro", max_timepoints: Optional[int] = None) -> Dict[str, Any]:
    """
    Load LFP from processing['ecephys'][which].
    Returns dict with signals, fs, meta.
    """
    ece = nwbfile.processing["ecephys"]
    obj = ece[which]  # likely LFP container with .electrical_series
    # For NWB LFP, the real signals are usually in obj.electrical_series (dict)
    es_dict = getattr(obj, "electrical_series", None)
    if es_dict is None or len(es_dict) == 0:
        # fallback: treat obj itself as timeseries-like
        signals = _read_timeseries(obj, max_timepoints=max_timepoints)
        fs = getattr(obj, "rate", None)
        return {"signals": signals, "fs": fs, "series_key": which, "meta": {"container": which, "note": "fallback"}}

    # pick first electrical series by default (or you can pick by name)
    es_key = list(es_dict.keys())[0]
    ts = es_dict[es_key]

    signals = _read_timeseries(ts, max_timepoints=max_timepoints)
    fs = getattr(ts, "rate", None)

    meta = {
        "container": which,
        "electrical_series_key": es_key,
        "unit": getattr(ts, "unit", None),
        "starting_time": getattr(ts, "starting_time", None),
        "has_timestamps": getattr(ts, "timestamps", None) is not None,
        "available_series": list(es_dict.keys()),
    }
    return {"signals": signals, "fs": fs, "meta": meta}


def load_nwb_behavior(nwbfile, key: str, max_timepoints: Optional[int] = None) -> Dict[str, Any]:
    """
    Load one behavior item from processing['behavior'][key]
    key in: 'EyeTracking','PupilTracking','Blink','Saccade','Fixation'
    Returns a dict of series_name -> {data, shape, fs,...} or a simple timeseries fallback.
    """
    beh = nwbfile.processing["behavior"]
    obj = beh[key]

    out: Dict[str, Any] = {"_container": key}

    # Many behavior containers are TimeSeries-like or have multiple series inside
    # Common patterns: obj.time_series or obj.spatial_series (dict-like)
    ts_dict = None
    for attr in ("time_series", "spatial_series", "behavioral_time_series"):
        d = getattr(obj, attr, None)
        if d is not None and hasattr(d, "keys"):
            ts_dict = (attr, d)
            break

    if ts_dict is not None:
        attr_name, d = ts_dict
        out["_series_attr"] = attr_name
        out["_series_keys"] = list(d.keys())
        for name in d.keys():
            ts = d[name]
            data = _read_timeseries(ts, max_timepoints=max_timepoints)
            out[name] = {
                "data": data,
                "shape": data.shape,
                "fs": getattr(ts, "rate", None),
                "unit": getattr(ts, "unit", None),
                "starting_time": getattr(ts, "starting_time", None),
                "has_timestamps": getattr(ts, "timestamps", None) is not None,
            }
        return out

    # Fallback: treat obj itself as timeseries-like
    if hasattr(obj, "data"):
        data = _read_timeseries(obj, max_timepoints=max_timepoints)
        out["data"] = data
        out["shape"] = data.shape
        out["fs"] = getattr(obj, "rate", None)
        out["unit"] = getattr(obj, "unit", None)
        out["starting_time"] = getattr(obj, "starting_time", None)
        out["has_timestamps"] = getattr(obj, "timestamps", None) is not None
        return out

    # Unknown structure
    out["error"] = f"Unsupported behavior object type: {type(obj)}"
    return out


def load_nwb_stimulus_movieframe_time(nwbfile, max_timepoints: Optional[int] = None) -> Dict[str, Any]:
    stim = nwbfile.stimulus
    if "movieframe_time" not in stim:
        return {"error": "movieframe_time not found in nwbfile.stimulus"}
    ts = stim["movieframe_time"]
    data = _read_timeseries(ts, max_timepoints=max_timepoints)
    return {
        "data": data,
        "shape": data.shape,
        "fs": getattr(ts, "rate", None),
        "unit": getattr(ts, "unit", None),
        "starting_time": getattr(ts, "starting_time", None),
        "has_timestamps": getattr(ts, "timestamps", None) is not None,
    }


# -----------------------
# BIDS helpers
# -----------------------
def build_bids_layout(bids_root: Union[str, Path]) -> BIDSLayout:
    bids_root = Path(bids_root)
    if not bids_root.exists():
        raise FileNotFoundError(f"BIDS root not found: {bids_root}")
    return BIDSLayout(str(bids_root), validate=False)


def get_bold_files_for_subject(layout: BIDSLayout, bids_sub: str) -> List[str]:
    return layout.get(
        subject=bids_sub,
        datatype="func",
        suffix="bold",
        extension=["nii", "nii.gz"],
        return_type="file",
    )


def load_bold_headers(bold_files: List[str]) -> List[Dict[str, Any]]:
    out = []
    for f in bold_files:
        img = nib.load(f)  # header fast
        zooms = img.header.get_zooms()
        tr = zooms[3] if len(zooms) >= 4 else None
        out.append(
            {
                "path": f,
                "shape": img.shape,
                "voxel_size_xyz": zooms[:3] if len(zooms) >= 3 else None,
                "TR": tr,
            }
        )
    return out


# -----------------------
# Main loader (int subjects)
# -----------------------
def load_selected_subjects(
    sub_nums: List[int],
    nwb_root: Union[str, Path],
    bids_root: Union[str, Path],
    *,
    load_lfp: bool = True,
    lfp_kind: str = "LFP_macro",  # or "LFP_micro"
    load_eye: bool = True,
    eye_keys: List[str] = None,   # default loads all known
    load_fmri: bool = True,
    max_nwb_timepoints: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:

    if eye_keys is None:
        eye_keys = ["EyeTracking", "PupilTracking", "Blink", "Saccade", "Fixation"]

    layout = build_bids_layout(bids_root)

    out: Dict[int, Dict[str, Any]] = {}
    for sub in sub_nums:
        bids_sub = bids_subject_from_int(sub)
        nwb_sub = nwb_subject_from_int(sub)

        subj_obj: Dict[str, Any] = {"meta": {"sub_int": sub, "bids_sub": bids_sub, "nwb_sub": nwb_sub}}

        # ---- NWB ----
        try:
            nwb_path = find_nwb_file_for_subject(nwb_root, nwb_sub)
            subj_obj["meta"]["nwb_path"] = str(nwb_path)

            io = NWBHDF5IO(str(nwb_path), "r", load_namespaces=True)
            nwbfile = io.read()

            if load_lfp:
                subj_obj["lfp"] = load_nwb_lfp(nwbfile, which=lfp_kind, max_timepoints=max_nwb_timepoints)

            if load_eye:
                eye_dict = {}
                for k in eye_keys:
                    try:
                        eye_dict[k] = load_nwb_behavior(nwbfile, k, max_timepoints=max_nwb_timepoints)
                    except Exception as e:
                        eye_dict[k] = {"error": repr(e)}
                subj_obj["eye"] = eye_dict

            # stimulus timing (useful for alignment)
            subj_obj["movieframe_time"] = load_nwb_stimulus_movieframe_time(nwbfile, max_timepoints=max_nwb_timepoints)

            io.close()

        except Exception as e:
            subj_obj["nwb_error"] = repr(e)

        # ---- fMRI (BIDS) ----
        if load_fmri:
            try:
                bold_files = get_bold_files_for_subject(layout, bids_sub)
                subj_obj["fmri"] = {
                    "bold_files": bold_files,
                    "bold_headers": load_bold_headers(bold_files),
                }
            except Exception as e:
                subj_obj["fmri_error"] = repr(e)

        out[sub] = subj_obj

    return out


def summarize_loaded(data: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for sub, obj in data.items():
        row: Dict[str, Any] = {"sub": sub}

        # LFP
        lfp = obj.get("lfp", None)
        if isinstance(lfp, dict) and "signals" in lfp:
            sig = lfp["signals"]
            row["lfp_shape"] = sig.shape
            row["lfp_fs"] = lfp.get("fs", None)
        else:
            row["lfp_shape"] = None
            if "nwb_error" in obj:
                row["nwb_error"] = obj["nwb_error"]

        # EyeTracking summary: just list what keys exist + one series shape
        eye = obj.get("eye", {})
        if isinstance(eye, dict) and len(eye) > 0:
            row["eye_keys"] = ",".join(list(eye.keys()))
            # try to find one shape
            any_shape = None
            try:
                et = eye.get("EyeTracking", {})
                # find first real series
                for k, v in et.items():
                    if isinstance(v, dict) and "shape" in v:
                        any_shape = v["shape"]
                        break
                if any_shape is None and isinstance(et, dict) and "shape" in et:
                    any_shape = et["shape"]
            except Exception:
                pass
            row["eye_any_shape"] = any_shape
        else:
            row["eye_keys"] = None
            row["eye_any_shape"] = None

        # fMRI
        fmri = obj.get("fmri", {})
        headers = fmri.get("bold_headers", []) if isinstance(fmri, dict) else []
        row["n_bold_runs"] = len(headers) if headers else 0
        if headers:
            row["bold0_shape"] = headers[0].get("shape")
            row["bold0_TR"] = headers[0].get("TR")
        if "fmri_error" in obj:
            row["fmri_error"] = obj["fmri_error"]

        rows.append(row)

    return pd.DataFrame(rows)