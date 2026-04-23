import os
from pathlib import Path

from tqdm import tqdm

import rename
from project import analyze_and_save_dice
from settings import Settings


def main():
    settings = Settings.from_file(Path("settings.toml"))

    photos = rename.rename(settings.photos, "png")
    os.makedirs(settings.output, exist_ok=True)

    iterator = tqdm(photos, total=len(photos), desc="Processing photos")
    for item in iterator:
        output = analyze_and_save_dice(
            item,
            settings,
        )
        iterator.set_postfix(
            {"Number of dices": output.num_dice, "Number of pips": output.num_pips}
        )


if __name__ == "__main__":
    main()
