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


def is_numeric(s: str):
    """is_numeric returns true for anything convertible to a float from strings."""
    s = s.replace(",", "")
    try:
        _ = float(s)
        return True
    except:
        return False


def is_time(s: str):
    """is_time returns true for time like strings, with a wide, inaccurate berth."""
    if ":" in s:
        return is_numeric(s.replace(":", ""))


# TODO: there is existing normalization code in the transformer for
# normalization.
def normalize_en_line(line: str) -> str:

    line = line.lower()

    # Bespoke rules, based on eyeballing the output lines. Aren't, can't, doesn't, isn't, won't...
    line = (
        line.replace("u.s.", "united states")
        .replace("u.k.", "united kingdom")
        .replace(" l.a.", " los angeles")
        .replace("d.c.", "district of columbia")
        .replace("won't", "will not")
        .replace("'s", "")
        .replace("n't", " not")
    )

    lower_tokens = line.split()
    for i, token in enumerate(lower_tokens):
        if token[0] == "$":
            lower_tokens[i] = "<dollars>"
            continue
        if token[-1] == "%":
            lower_tokens[i] = "<percentage>"
            continue
        if is_numeric(token):
            lower_tokens[i] = "<number>"
            continue
        if is_time(token):
            lower_tokens[i] = "<datetime>"
            continue

    mappings = [
        (".", " "),
        ("!", " "),
        ("?", " "),
        (",", " "),
        (":", " "),
        (";", " "),
        ('"', " "),
        ("'", ""),
        (")", ""),
        ("(", ""),
        ("-", " "),
        ("—", " "),
        ("*", ""),
    ]
    trans_table = {}
    for tup in mappings:
        trans_table[tup[0]] = tup[1]
    trans_table = str.maketrans(trans_table)

    line = " ".join(lower_tokens).translate(trans_table).strip()

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

    line_hashes = set()
    for line in lines:
        line_hash = hash(line)
        if line_hash not in line_hashes:
            print(line)
            line_hashes.add(line_hash)


if __name__ == "__main__":
    main()
