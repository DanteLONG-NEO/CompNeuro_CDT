import numpy as np
import pandas as pd
from collections import Counter

def polygon_area(coords):
    """
    Shoelace formula
    coords: array-like, shape [N, 2]
    """
    pts = np.asarray(coords, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        return 0.0

    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def parse_frame_idx(frame_key):
    """
    'frame_118' -> 118
    """
    try:
        return int(str(frame_key).split("_")[-1])
    except Exception:
        return np.nan
    
def one_hot_face_features(df_face):
    df = df_face.copy()

    # dominant_emotion one-hot
    emo_dummies = pd.get_dummies(df["dominant_emotion"], prefix="emo")

    df = pd.concat([df, emo_dummies], axis=1)
    return df


def build_face_feature_df(
    face_annots,
    target_id="Jacky",
    front_labels=("front",),
    emotion_fill="none"
):
    """
    Parameters
    ----------
    face_annots : dict
        Example:
        {
            'frame_118': {
                'p0': {'coords': ..., 'ID': 'Jacky', 'hpose': 'R90', 'emotion': 'neutral'},
                'p1': {...}
            },
            ...
        }

    target_id : str
        Person identity for *_present feature, e.g. 'Jacky'

    front_labels : tuple[str]
        Which hpose labels count as frontal face

    emotion_fill : str
        Value used when no face exists in a frame

    Returns
    -------
    df : pd.DataFrame
        Columns:
        [
            'frame_key',
            'frame_idx',
            'n_faces',
            'total_face_area',
            'max_face_area',
            'Jacky_present',
            'any_front_face',
            'dominant_emotion',
        ]
    """
    rows = []

    # sort by frame index if possible
    sorted_items = sorted(face_annots.items(), key=lambda kv: parse_frame_idx(kv[0]))

    for frame_key, frame_data in sorted_items:
        frame_idx = parse_frame_idx(frame_key)

        # frame_data should be like {'p0': {...}, 'p1': {...}}
        if not isinstance(frame_data, dict):
            frame_data = {}

        face_areas = []
        emotions = []
        any_front_face = 0
        target_present = 0
        n_faces = 0

        for _, face_info in frame_data.items():
            if not isinstance(face_info, dict):
                continue

            coords = face_info.get("coords", None)
            face_id = face_info.get("ID", None)
            hpose = face_info.get("hpose", None)
            emotion = face_info.get("emotion", None)

            area = polygon_area(coords) if coords is not None else 0.0

            face_areas.append(area)
            n_faces += 1

            if face_id == target_id:
                target_present = 1

            if hpose in front_labels:
                any_front_face = 1

            if emotion is not None:
                emotions.append(str(emotion))

        total_face_area = float(np.sum(face_areas)) if len(face_areas) > 0 else 0.0
        max_face_area = float(np.max(face_areas)) if len(face_areas) > 0 else 0.0

        if len(emotions) > 0:
            dominant_emotion = Counter(emotions).most_common(1)[0][0]
        else:
            dominant_emotion = emotion_fill

        rows.append({
            "frame_key": frame_key,
            "frame_idx": frame_idx,
            "movie_time": float(frame_idx / 25.0),
            "n_faces": int(n_faces),
            "total_face_area": total_face_area,
            "max_face_area": max_face_area,
            f"{target_id}_present": int(target_present),
            "any_front_face": int(any_front_face),
            "dominant_emotion": dominant_emotion,
        })

    
    df = pd.DataFrame(rows).sort_values("frame_idx").reset_index(drop=True)
    df = one_hot_face_features(df)
    return df