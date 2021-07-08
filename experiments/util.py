from pathlib import Path


def clean_dir(dirpath: Path):
    # Delete everything in a dir, recursively.
    for item in dirpath.iterdir():
        if item.is_dir():
            clean_dir(item)
            item.rmdir()
        else:
            item.unlink()
