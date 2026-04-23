import os
from pathlib import Path
import re

from tqdm import tqdm


def rename(path: Path, suffix: str) -> list[Path]:
    paths = sorted(list(path.glob(f"*.{suffix}")))

    pattern = re.compile(rf"^\d+\.{suffix}$")
    if all(pattern.match(p.name) for p in paths):
        return paths

    for index, item in tqdm(enumerate(paths), total=len(paths), desc="Renaming photos"):
        new = item.with_name(f"{index}.{suffix}")
        os.rename(item, new)
        paths[index] = new

    return paths
