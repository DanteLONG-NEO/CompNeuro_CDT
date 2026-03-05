"""
Data summary for bmovie NWB + BIDS dataset.

Usage (Notebook):
    - Just run the whole cell.

Usage (CLI):
    python data_summary.py --nwb_dir "..." --bids_dir "..." --out_csv "summary.csv"

Requirements:
    pip install pynwb pybids nibabel pandas numpy
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---- Optional imports with friendly error messages ----
try:
    from pynwb import NWBHDF5IO
except Exception as e:
    NWBHDF5IO = None
    _pynwb_err = e

try:
    import nibabel as nib
except Exception as e:
    nib = None
    _nib_err = e

try:
    from bids import BIDSLayout
except Exception as e:
    BIDSLayout = None
    _pybids_err = e


def _as_str_path(p: os.PathLike | str) -> str:
    return str(p)


def _safe_getattr(obj: Any, attr: str, default=None):
    try:
        return getattr(obj, attr)
    except Exception:
        return default


# ----------------------------
# NWB summary
# ----------------------------
def find_nwb_files(nwb_dir: str | Path) -> List[Path]:
    nwb_dir = Path(nwb_dir)
    if not nwb_dir.exists():
        raise FileNotFoundError(f"NWB directory not found: {nwb_dir}")
    # Recursively find *.nwb
    files = sorted(nwb_dir.rglob("*.nwb"))
    return files


def summarize_one_nwb(nwb_path: Path) -> Dict[str, Any]:
    if NWBHDF5IO is None:
        raise RuntimeError(
            f"pynwb not available: {_pynwb_err}\nInstall with: pip install pynwb"
        )

    row: Dict[str, Any] = {
        "type": "NWB",
        "file": nwb_path.name,
        "path": str(nwb_path),
    }

    io = NWBHDF5IO(str(nwb_path), "r", load_namespaces=True)
    try:
        nwb = io.read()

        # Basic session/subject metadata (best-effort)
        row["session_description"] = _safe_getattr(nwb, "session_description", None)
        row["identifier"] = _safe_getattr(nwb, "identifier", None)
        row["session_start_time"] = str(_safe_getattr(nwb, "session_start_time", None))

        subj = _safe_getattr(nwb, "subject", None)
        if subj is not None:
            row["subject_id"] = _safe_getattr(subj, "subject_id", None)
            row["age"] = _safe_getattr(subj, "age", None)
            row["sex"] = _safe_getattr(subj, "sex", None)
            row["species"] = _safe_getattr(subj, "species", None)
        else:
            row["subject_id"] = None
            row["age"] = None
            row["sex"] = None
            row["species"] = None

        # Electrodes table info
        try:
            electrodes_df = nwb.electrodes.to_dataframe()
            row["n_electrodes"] = int(electrodes_df.shape[0])
        except Exception:
            row["n_electrodes"] = None

        # Acquisition series summary
        acq = _safe_getattr(nwb, "acquisition", None)
        acq_keys = list(acq.keys()) if acq is not None else []
        row["acquisition_keys"] = ",".join(acq_keys)

        # Collect per-series details into a compact string
        series_summaries = []
        for key in acq_keys:
            ts = acq[key]
            # Shape without loading full data
            shape = None
            try:
                shape = tuple(ts.data.shape)  # h5py dataset shape
            except Exception:
                shape = None

            # Sampling rate if present
            rate = _safe_getattr(ts, "rate", None)
            starting_time = _safe_getattr(ts, "starting_time", None)
            unit = _safe_getattr(ts, "unit", None)

            # Some series have timestamps; don't load, just presence + length
            has_timestamps = _safe_getattr(ts, "timestamps", None) is not None
            ts_len = None
            try:
                ts_len = int(ts.data.shape[0]) if shape is not None else None
            except Exception:
                ts_len = None

            series_summaries.append(
                f"{key} shape={shape} len={ts_len} rate={rate} unit={unit} "
                f"has_timestamps={has_timestamps} starting_time={starting_time}"
            )

        row["acquisition_summary"] = " | ".join(series_summaries)

    finally:
        io.close()

    return row


def summarize_nwb_dir(nwb_dir: str | Path, max_files: Optional[int] = None) -> pd.DataFrame:
    files = find_nwb_files(nwb_dir)
    if max_files is not None:
        files = files[:max_files]

    rows = []
    for f in files:
        try:
            rows.append(summarize_one_nwb(f))
        except Exception as e:
            rows.append(
                {
                    "type": "NWB",
                    "file": f.name,
                    "path": str(f),
                    "error": repr(e),
                }
            )
    return pd.DataFrame(rows)


# ----------------------------
# BIDS summary
# ----------------------------
def summarize_one_nifti(nifti_path: str | Path) -> Dict[str, Any]:
    if nib is None:
        raise RuntimeError(
            f"nibabel not available: {_nib_err}\nInstall with: pip install nibabel"
        )

    p = Path(nifti_path)
    img = nib.load(str(p))  # header load (fast); data not loaded
    hdr = img.header

    shape = img.shape
    zooms = hdr.get_zooms()  # voxel size + TR if 4D
    tr = zooms[3] if len(zooms) >= 4 else None
    voxel = zooms[:3] if len(zooms) >= 3 else None

    return {
        "type": "NIFTI",
        "file": p.name,
        "path": str(p),
        "shape": shape,
        "voxel_size_xyz": voxel,
        "TR": tr,
    }


def summarize_bids_root(bids_dir: str | Path, max_files_each: int = 10) -> pd.DataFrame:
    if BIDSLayout is None:
        raise RuntimeError(
            f"pybids not available: {_pybids_err}\nInstall with: pip install pybids"
        )
    bids_dir = Path(bids_dir)
    if not bids_dir.exists():
        raise FileNotFoundError(f"BIDS directory not found: {bids_dir}")

    layout = BIDSLayout(str(bids_dir), validate=False)

    rows: List[Dict[str, Any]] = []

    # Basic dataset-level info
    rows.append(
        {
            "type": "BIDS_ROOT",
            "path": str(bids_dir),
            "subjects": ",".join(layout.get_subjects() or []),
            "n_subjects": len(layout.get_subjects() or []),
            "tasks": ",".join(layout.get_tasks() or []),
            "sessions": ",".join(layout.get_sessions() or []),
        }
    )

    # Functional bold
    bold_files = layout.get(
        datatype="func", suffix="bold", extension=["nii", "nii.gz"], return_type="file"
    )
    rows.append(
        {
            "type": "BIDS_INDEX",
            "kind": "bold",
            "n_files": len(bold_files),
        }
    )
    for f in bold_files[:max_files_each]:
        try:
            r = summarize_one_nifti(f)
            r["kind"] = "bold"
            rows.append(r)
        except Exception as e:
            rows.append({"type": "NIFTI", "kind": "bold", "path": str(f), "error": repr(e)})

    # Anatomical T1w
    t1w_files = layout.get(
        datatype="anat", suffix="T1w", extension=["nii", "nii.gz"], return_type="file"
    )
    rows.append(
        {
            "type": "BIDS_INDEX",
            "kind": "T1w",
            "n_files": len(t1w_files),
        }
    )
    for f in t1w_files[:max_files_each]:
        try:
            r = summarize_one_nifti(f)
            r["kind"] = "T1w"
            rows.append(r)
        except Exception as e:
            rows.append({"type": "NIFTI", "kind": "T1w", "path": str(f), "error": repr(e)})

    return pd.DataFrame(rows)


# ----------------------------
# Combined summary runner
# ----------------------------
def run_summary(nwb_dir: str | Path, bids_dir: str | Path,
                max_nwb_files: Optional[int] = None,
                max_bids_files_each: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nwb_df = summarize_nwb_dir(nwb_dir, max_files=max_nwb_files)
    bids_df = summarize_bids_root(bids_dir, max_files_each=max_bids_files_each)
    return nwb_df, bids_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nwb_dir", required=True, help="Directory containing NWB files (recursive).")
    parser.add_argument("--bids_dir", required=True, help="BIDS root directory.")
    parser.add_argument("--max_nwb", type=int, default=None, help="Max number of NWB files to summarize.")
    parser.add_argument("--max_bids_each", type=int, default=10, help="Max NIfTI files to list per kind (bold/T1w).")
    parser.add_argument("--out_csv", default=None, help="If set, write combined summary CSV to this path.")
    args = parser.parse_args()

    nwb_df, bids_df = run_summary(
        nwb_dir=args.nwb_dir,
        bids_dir=args.bids_dir,
        max_nwb_files=args.max_nwb,
        max_bids_files_each=args.max_bids_each,
    )

    print("\n===== NWB summary (head) =====")
    print(nwb_df.head())

    print("\n===== BIDS summary (head) =====")
    print(bids_df.head())

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        combined = pd.concat(
            [
                nwb_df.assign(_table="NWB"),
                bids_df.assign(_table="BIDS"),
            ],
            ignore_index=True,
        )
        combined.to_csv(out, index=False)
        print(f"\nWrote combined summary CSV to: {out}")


if __name__ == "__main__":
    main()