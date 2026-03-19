import csv
import argparse
from typing import List, Tuple

def extract_scene_times(csv_path: str) -> List[Tuple[int, float, float, float]]:
    """Return list of (scene_id, start_t, end_t, duration) sorted by scene_id."""
    scenes = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid_raw = row.get('scene_id')
            if sid_raw is None or sid_raw == '':
                continue
            try:
                sid = int(float(sid_raw))
            except Exception:
                sid = sid_raw

            try:
                start = float(row.get('shot_start_t') or 0.0)
            except Exception:
                start = 0.0

            dur_raw = row.get('shot_dur_t')
            try:
                dur = float(dur_raw) if (dur_raw is not None and dur_raw != '') else 0.0
            except Exception:
                dur = 0.0

            end = start + dur

            if sid not in scenes:
                scenes[sid] = {'start': start, 'end': end}
            else:
                scenes[sid]['start'] = min(scenes[sid]['start'], start)
                scenes[sid]['end'] = max(scenes[sid]['end'], end)

    out = []
    for sid in sorted(scenes, key=lambda x: float(x)):
        s = scenes[sid]
        out.append((sid, s['start'], s['end'], s['end'] - s['start']))
    return out


def main():
    p = argparse.ArgumentParser(description='Extract scene time ranges from scenecut CSV')
    p.add_argument('csv', help='path to scenecut_info.csv')
    p.add_argument('--out', help='optional output CSV to save ranges')
    args = p.parse_args()

    ranges = extract_scene_times(args.csv)
    for sid, start, end, dur in ranges:
        print(f"{sid},{start:.3f},{end:.3f},{dur:.3f}")

    if args.out:
        with open(args.out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['scene_id','start_t','end_t','duration_t'])
            for sid, start, end, dur in ranges:
                writer.writerow([sid, f"{start:.3f}", f"{end:.3f}", f"{dur:.3f}"])


if __name__ == '__main__':
    main()
