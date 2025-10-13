from pathlib import Path



def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _to_str_list(x):
    out = []
    for v in x:
        if isinstance(v, bytes):
            out.append(v.decode("utf-8", errors="ignore"))
        else:
            out.append(str(v))
    return out
