
from pathlib import Path

def mkdir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)