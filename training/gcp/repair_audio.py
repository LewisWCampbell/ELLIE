"""Drop episodes whose assembled audio was damaged by a Spot preemption.

A preemption is a hard power-off. A file written just before it can survive as
zero bytes even though it was renamed into place, because its data had not
reached the disk yet. `prepare.sh mimi` then fails on it ("Expecting value:
line 1 column 1"). The clips those files were built from are already deleted,
so instead of rebuilding them this drops the few affected episodes from
retimed.jsonl (a backup is kept) and deletes their broken files.

    python3 training/gcp/repair_audio.py ~/elle-work/data/retimed.jsonl ~/elle-work/audio
"""

import json
import os
import shutil
import sys
import wave


def broken(audio_dir: str, ep_id: str) -> str:
    wav = os.path.join(audio_dir, ep_id + ".wav")
    js = os.path.join(audio_dir, ep_id + ".json")
    try:
        with wave.open(wav, "rb") as w:
            if w.getnframes() == 0 or w.getnchannels() != 2:
                return "empty or wrong wav"
            # The header can be intact over zeroed data; check the tail is readable.
            w.setpos(w.getnframes() - 1)
            if len(w.readframes(1)) != 4:
                return "truncated wav"
    except Exception as e:  # noqa: BLE001
        return f"unreadable wav ({type(e).__name__})"
    try:
        with open(js, encoding="utf-8") as f:
            a = json.load(f)
        if a.get("episode_id") != ep_id:
            return "alignment for another episode"
    except Exception as e:  # noqa: BLE001
        return f"unreadable alignment ({type(e).__name__})"
    return ""


def main(retimed: str, audio_dir: str) -> int:
    keep, bad = [], []
    with open(retimed, encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    for line in lines:
        ep_id = json.loads(line)["id"]
        why = broken(audio_dir, ep_id)
        (bad.append((ep_id, why)) if why else keep.append(line))
    for ep_id, why in bad:
        print(f"dropping {ep_id}: {why}")
        for ext in (".wav", ".json"):
            p = os.path.join(audio_dir, ep_id + ext)
            if os.path.exists(p):
                os.remove(p)
    if bad:
        backup = retimed + ".before_repair"
        if not os.path.exists(backup):
            shutil.copy2(retimed, backup)
        tmp = retimed + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(keep)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, retimed)
    print(f"checked {len(lines)} episodes: {len(bad)} dropped, {len(keep)} kept")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
