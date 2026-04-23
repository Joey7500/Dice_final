import tomllib
from dataclasses import dataclass
from pathlib import Path
from pprint import pp
from typing import Self


@dataclass
class Settings:
    scale_percent: float
    min_die_area: int
    expected_die_area: int
    adaptive_block: int
    adaptive_c: int
    min_pip_blob_area: int
    max_pip_blob_area: int
    expected_pip_area: int
    pip_radius_est: int
    dist_thresh_abs_multiplier: float

    photos: Path
    output: Path

    @property
    def dist_thresh_abs(self) -> float:
        return self.pip_radius_est * self.dist_thresh_abs_multiplier

    @classmethod
    def from_file(cls, path: Path) -> Self:
        with open(path, "rb") as file:
            data = tomllib.load(file)

            data["photos"] = Path(data["photos"])
            data["output"] = Path(data["output"])

            return cls(**data)


if __name__ == "__main__":
    settings = Settings.from_file(Path("settings.toml"))
    pp(settings)
