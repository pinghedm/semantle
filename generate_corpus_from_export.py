"""module docstring"""
import os
import json


EXPORTS_DIR = "exports"
# would just running jq to extract contents be better?
# sure aint faster!
for _, __, files in os.walk(EXPORTS_DIR):
    for file in files:
        if not file.endswith(".json"):
            continue
        with open(f"{EXPORTS_DIR}/{file}", encoding="utf8") as f, open(
            f"{EXPORTS_DIR}/{file}.content", "w", encoding="utf8"
        ) as g:
            parsed_file = json.load(f)
            contents = [m["content"] for m in parsed_file["messages"]]
            json.dump(contents, g)
