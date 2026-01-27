"""
fb_data_to_lines extracts the 'description' og field from my old fbData.py
definition, for which each line is a tuple whose [0] is the link and [1] is the
og_object from facebook.

Note this is barely useful: each fbdata file is a tuple whose [1] member is a json
object

Env vars:
    * FB_DATA_PATH: required, path to the

"""

import os
from pathlib import Path
from typing import Iterator, Callable


ENV_FB_DATA_PATH = "FB_DATA_PATH"


def get_lines(lines_path: Path) -> Iterator[str]:
    with open(lines_path, "r", encoding="utf8") as ifile:
        for line in ifile:
            line = line.strip()
            if not line:
                continue
            yield line


def parse_fb_data_lines(line_gen: Iterator) -> Iterator[str]:
    for line in line_gen:
        tup = eval(line)
        fb_object = tup[1]
        if "og_object" in fb_object:
            og_object = fb_object["og_object"]
            # Just grab the description text. We could get more, this is simple
            # for now.
            if "description" in og_object:
                yield og_object["description"]


# TODO: there is existing normalization code in the transformer for
# normalization.
def normalize_en_line(line: str) -> str:
    line = (
        line.lower()
        .replace(".", " ")
        .replace(",", " ")
        .replace(":", " ")
        .replace('"', " ")
        .strip()
    )

    while len(line) > len(line.replace("  ", " ")):
        line = line.replace("  ", " ")

    return line


def normalize_lines(
    line_gen: Iterator, normalize_fn: Callable[[str], str]
) -> Iterator[str]:
    for line in line_gen:
        yield normalize_fn(line)


def main():
    if ENV_FB_DATA_PATH not in os.environ:
        print(f"{ENV_FB_DATA_PATH} required but not found: ", os.environ)
        exit(1)

    fb_data_path = Path(os.environ[ENV_FB_DATA_PATH])
    if not fb_data_path.exists():
        print(f"{fb_data_path} not found")
        exit(1)

    lines = get_lines(fb_data_path)
    fb_lines = parse_fb_data_lines(lines)
    lines = normalize_lines(fb_lines, normalize_en_line)
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
