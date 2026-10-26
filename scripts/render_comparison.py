"""Create a reproducible 1600px comparison PNG from SVGs and local Git history.

Requires a Chromium browser (Edge/Chrome); no model downloads or Python imaging
dependencies. Top-screenful panels fit the same width without vertical compression.
The detail panels show individual node excerpts at a shared, enlarged scale.
"""

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from html import escape
from pathlib import Path
from xml.etree import ElementTree as ET

NS = {"s": "http://www.w3.org/2000/svg"}
ET.register_namespace("", NS["s"])
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
REPO = Path(__file__).resolve().parents[1]
PANEL_WIDTH = 736
SCREENFUL_HEIGHT = PANEL_WIDTH * 10 // 16
DETAIL_HEIGHT = 375


def historical_svg(revision: str) -> ET.Element:
    result = subprocess.run(
        ["git", "show", f"{revision}:examples/svg/tinyllama_after.svg"],
        cwd=REPO,
        capture_output=True,
        check=True,
    )
    return ET.fromstring(result.stdout)


def svg_panel(root: ET.Element, height: int, *, top_screenful: bool = False) -> str:
    if top_screenful:
        x, y, width, _ = map(float, root.attrib["viewBox"].split())
        # Keep the full width and crop the height to a window with the same scale.
        root.set("viewBox", f"{x} {y} {width} {width * height / PANEL_WIDTH}")
    root.set("width", str(PANEL_WIDTH))
    root.set("height", str(height))
    root.set("preserveAspectRatio", "xMidYMin meet")
    return ET.tostring(root, encoding="unicode")


def detail(root: ET.Element, ids: set[str]) -> tuple[ET.Element, tuple[float, ...]]:
    root = copy.deepcopy(root)
    group = root.find("s:g", NS)
    tx, ty = map(
        float, re.search(r"translate\(([^)]+)\)", group.attrib["transform"]).group(1).split()
    )
    points = []
    for child in list(group):
        kind = child.attrib.get("class")
        keep = False
        if kind == "node" and child.attrib["id"] in ids:
            keep = True
            polygon = child.find(".//s:polygon", NS)
            points.extend(
                (float(x) + tx, float(y) + ty)
                for x, y in (p.split(",") for p in polygon.attrib["points"].split())
            )
        elif kind == "edge":
            endpoints = child.find("s:title", NS).text.split("->")
            keep = all("node_" + endpoint.split(":")[0] in ids for endpoint in endpoints)
        if not keep:
            group.remove(child)
    if not points:
        raise ValueError("Requested detail nodes are missing from SVG.")
    return root, (
        min(x for x, _ in points),
        min(y for _, y in points),
        max(x for x, _ in points),
        max(y for _, y in points),
    )


def find_browser(explicit: str | None) -> str:
    candidates = (
        [explicit]
        if explicit
        else [
            shutil.which(name)
            for name in ("msedge", "chromium", "chromium-browser", "google-chrome")
        ]
    )
    if not explicit and os.name == "nt":
        candidates += [
            str(Path(os.environ.get(variable, "")) / tail)
            for variable in ("PROGRAMFILES(X86)", "PROGRAMFILES")
            for tail in (
                "Microsoft/Edge/Application/msedge.exe",
                "Google/Chrome/Application/chrome.exe",
            )
        ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise FileNotFoundError("Set --browser to a Chromium/Edge/Chrome executable.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", help="path to Edge, Chromium or Chrome")
    parser.add_argument("--previous-revision", default="04b4289")
    parser.add_argument("--original-revision", default="028ef46")
    parser.add_argument(
        "--current-svg", type=Path, default=REPO / "examples/svg/tinyllama_after.svg"
    )
    parser.add_argument("--output", type=Path, default=REPO / "examples/readability-comparison.png")
    args = parser.parse_args()
    previous = historical_svg(args.previous_revision)
    original = historical_svg(args.original_revision)
    current = ET.parse(args.current_svg).getroot()
    before_width = float(previous.attrib["viewBox"].split()[2])
    after_width = float(current.attrib["viewBox"].split()[2])
    metadata = json.loads(args.current_svg.with_suffix(".json").read_text(encoding="utf-8"))
    closeups = [
        [detail(root, {metadata["nodes"][node_id]["svg_id"]}) for root in (original, current)]
        for node_id in ("42", "43", "45")
    ]
    crop_width = max(box[2] - box[0] for pair in closeups for _, box in pair) + 24
    crop_height = max(box[3] - box[1] for pair in closeups for _, box in pair) + 24
    details = [[], []]
    for closeup in closeups:
        for column, (root, box) in enumerate(closeup):
            root.set("viewBox", f"{box[0] - 12} {box[1] - 12} {crop_width} {crop_height}")
            details[column].append(svg_panel(root, DETAIL_HEIGHT))
    # All six node excerpts share crop dimensions and therefore the same scale.
    panels = [
        (
            f"Before — {before_width:,.0f} pt wide",
            svg_panel(previous, SCREENFUL_HEIGHT, top_screenful=True),
        ),
        (
            f"After — {after_width:,.0f} pt wide",
            svg_panel(copy.deepcopy(current), SCREENFUL_HEIGHT, top_screenful=True),
        ),
        ("Original labels: names and attributes", "".join(details[0])),
        ("Compact labels: details in JSON", "".join(details[1])),
    ]

    def pair(items):
        return (
            '<div class="pair">'
            + "".join(f"<section><h3>{escape(title)}</h3>{svg}</section>" for title, svg in items)
            + "</div>"
        )

    html = """<!doctype html><html><head><meta charset="utf-8"><style>
    * { box-sizing: border-box; } body { margin: 0; padding: 28px; width: 1600px;
    background: #f5f7fb; color: #17263c; font-family: Arial, sans-serif; }
    h1 { font-size: 30px; margin: 0 0 10px; } h2 { font-size: 23px; margin: 20px 0 8px; }
    h3 { font-size: 18px; margin: 0 0 12px; } p { font-size: 16px; margin: 6px 0 16px; }
    .pair { display:flex; gap:24px; } section { background:white; border:1px solid #d8dfeb;
    border-radius:8px; padding:11px; width:760px; overflow:hidden; }
    svg { display:block; } strong { color:#1558a4; }
    </style></head><body><h1>TinyLlama: readability improvements</h1>
    <h2>A. What you see when you open the SVG at fit-to-width</h2>"""
    html += (
        "<p>Same window width, top of the graph. "
        f"<strong>{before_width / after_width:.1f}× larger at fit-to-width.</strong></p>"
    )
    html += pair(panels[:2])
    html += (
        "<h2>B. The same three operations — equal scale</h2>"
        "<p>Convert #42 → MatMul #43 → Reshape #45. "
        "The red border is from the highlight pass.<br>"
        "Individual node excerpts; surrounding nodes and arrows omitted here. "
        "All connections remain in SVG/JSON.</p>"
    )
    html += pair(panels[2:]) + "</body></html>"
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = REPO / "artifacts/comparison"
    artifacts.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=artifacts, ignore_cleanup_errors=True) as temporary:
        folder = Path(temporary)
        page = folder / "comparison.html"
        page.write_text(html, encoding="utf-8")
        screenshot = folder / "comparison.png"
        subprocess.run(
            [
                find_browser(args.browser),
                "--headless",
                "--disable-gpu",
                "--no-first-run",
                "--disable-background-networking",
                "--disable-extensions",
                "--allow-file-access-from-files",
                f"--user-data-dir={folder / 'profile'}",
                f"--screenshot={screenshot}",
                "--window-size=1600,2040",
                "--force-device-scale-factor=1",
                "--hide-scrollbars",
                page.as_uri(),
            ],
            check=True,
            capture_output=True,
            timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        deadline = time.monotonic() + 10
        while not screenshot.is_file() and time.monotonic() < deadline:
            time.sleep(0.1)
        shutil.copyfile(screenshot, output)
    print(f"Created {output}")


if __name__ == "__main__":
    main()
