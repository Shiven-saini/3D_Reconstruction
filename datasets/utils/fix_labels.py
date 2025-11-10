#!/usr/bin/env python3
import argparse
from pathlib import Path

def relabel_file(path, old="0", new="1"):
    changed = False
    out_lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                out_lines.append(line)
                continue
            parts = s.split()
            # Only change the leading class token if it is exactly "0"
            if parts and parts[0] == old:
                parts[0] = new
                changed = True
                out_lines.append(" ".join(parts) + "\n")
            else:
                out_lines.append(line if line.endswith("\n") else line + "\n")
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(out_lines)
    return changed

def main():
    ap = argparse.ArgumentParser(description="Change YOLO class id 0 -> 1 in label .txt files")
    ap.add_argument("--root", type=Path, default=Path("."), help="Dataset root containing train/ valid/ test/ dirs")
    args = ap.parse_args()

    splits = ["train", "valid", "test"]
    total = changed = 0
    for split in splits:
        labels_dir = args.root / split / "labels"
        if not labels_dir.exists():
            continue
        for txt in labels_dir.rglob("*.txt"):
            total += 1
            if relabel_file(txt, "0", "1"):
                changed += 1
    print(f"Processed {total} files, modified {changed} files.")

if __name__ == "__main__":
    main()
