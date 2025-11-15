#!/usr/bin/env python3
"""Minimal unauthenticated Google Drive / Sheets downloader for PUBLIC files.

Supports:
1. Google Sheets (exports as CSV). Accepts optional gid parameter.
2. Regular Drive file share links (small/medium size) via uc export endpoint.
3. Raw file IDs or full share URLs.

No OAuth, no API client. Assumes: "Anyone with the link" access.

Usage examples:
    python scripts/gdrive_public.py --target https://docs.google.com/spreadsheets/d/FILE_ID/edit?gid=0
    python scripts/gdrive_public.py --target FILE_ID --sheet --gid 0
    python scripts/gdrive_public.py --target https://drive.google.com/file/d/FILE_ID/view?usp=sharing

Multiple targets:
    python scripts/gdrive_public.py --target <ID1> --target <ID2>

From a file list:
    python scripts/gdrive_public.py --file-list ids.txt

Environment variable (comma separated) also accepted:
    DRIVE_TARGETS="ID1,https://docs.google.com/spreadsheets/d/ID2/edit?gid=123" \
        python scripts/gdrive_public.py

Notes:
 - Large files that trigger Google virus scan warning may need confirmation token;
   basic handling included.
 - Sheets detection is automatic by URL pattern; force with --sheet if using raw ID.
"""

from __future__ import annotations
import os
import re
import argparse
import requests
from typing import List, Optional, Tuple

DRIVE_UC_BASE = "https://drive.google.com/uc?export=download&id="
SHEETS_EXPORT_TMPL = (
    "https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"
)
DEFAULT_OUTPUT_DIR = "data"


def parse_target(entry: str) -> Tuple[str, Optional[str], bool]:
    """Return (file_id, gid, is_sheet) from a raw ID or share URL."""
    entry = entry.strip()
    gid = None
    is_sheet = False

    # Sheets URL
    if "docs.google.com/spreadsheets" in entry:
        m = re.search(r"/d/([a-zA-Z0-9-_]+)", entry)
        if m:
            file_id = m.group(1)
        else:
            file_id = entry
        gid_m = re.search(r"[?&]gid=(\d+)", entry)
        if gid_m:
            gid = gid_m.group(1)
        is_sheet = True
        return file_id, gid, is_sheet

    # Generic file share URL
    if "drive.google.com/file" in entry:
        m = re.search(r"/file/d/([a-zA-Z0-9-_]+)", entry)
        if m:
            return m.group(1), None, False

    # Raw ID (heuristic: length and charset)
    file_id = entry.split("/")[0]
    return file_id, gid, is_sheet


def collect_targets(
    cli_targets: List[str], file_list: Optional[str]
) -> List[str]:
    targets: List[str] = []
    env_val = os.getenv("DRIVE_TARGETS")
    if env_val:
        targets.extend([t for t in env_val.split(",") if t.strip()])
    targets.extend(cli_targets)
    if file_list:
        try:
            with open(file_list, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        targets.append(line)
        except Exception as e:
            print(f"!!! Could not read file list {file_list}: {e}")
    return targets


def download_sheet(file_id: str, gid: Optional[str], out_dir: str) -> None:
    url = SHEETS_EXPORT_TMPL.format(file_id=file_id)
    if gid:
        url += f"&gid={gid}"
    out_name = f"{file_id}_sheet{gid or '0'}.csv"
    out_path = os.path.join(out_dir, out_name)
    try:
        print(f"-> Sheet CSV: {out_name}")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
    except Exception as e:
        print(f"!!! Sheet export failed for {file_id} gid={gid}: {e}")


def download_drive_file(file_id: str, out_dir: str) -> None:
    base_url = DRIVE_UC_BASE + file_id
    session = requests.Session()
    try:
        print(f"-> File: {file_id}")
        response = session.get(base_url, timeout=60, stream=True)

        # If Google shows an HTML confirmation (large files), try to extract
        # the confirm token either from cookies or from the HTML body and
        # re-request with the token.
        def parse_confirm_flow(resp):
            # 1) cookie-based token
            for key, val in resp.cookies.items():
                if key.startswith("download_warning"):
                    return (None, {"confirm": val})
            # 2) try to parse the HTML form action + hidden inputs
            try:
                text = resp.text
            except Exception:
                return (None, None)

            # Look for a form with hidden inputs (action + inputs)
            form_m = re.search(
                r"<form[^>]+action=\"([^\"]+)\"[^>]*>(.*?)</form>", text, re.S
            )
            if form_m:
                action = form_m.group(1)
                form_body = form_m.group(2)
                inputs = dict(
                    re.findall(
                        r"<input[^>]+name=\"([^\"]+)\"[^>]+value=\"([^\"]*)\"",
                        form_body,
                    )
                )
                return (action, inputs)

            # 3) fallback: look for confirm token in text
            m = re.search(
                r"name=\"?confirm\"?\s+value=\"?([0-9A-Za-z_\-]+)\"?", text
            )
            if m:
                return (None, {"confirm": m.group(1)})
            m2 = re.search(r"confirm=([0-9A-Za-z_\-]+)", text)
            if m2:
                return (None, {"confirm": m2.group(1)})
            return (None, None)

            action_url, params = parse_confirm_flow(response)
            if params or action_url:
                response.close()
                if action_url:
                    # Build the download URL from action and params
                    try:
                        from urllib.parse import urlencode

                        qs = urlencode(params)
                        download_url = action_url
                        if "?" in download_url:
                            download_url += "&" + qs
                        else:
                            download_url += "?" + qs
                    except Exception:
                        download_url = base_url
                else:
                    # fallback to uc endpoint with confirm token
                    download_url = base_url
                    if params and "confirm" in params:
                        download_url += f"&confirm={params['confirm']}"

                response = session.get(download_url, timeout=60, stream=True)

        response.raise_for_status()

        # Attempt to derive filename from headers
        cd = response.headers.get("Content-Disposition", "")
        name_match = re.search(r'filename="?([^";]+)"?', cd)
        if name_match:
            filename = name_match.group(1)
        else:
            filename = f"{file_id}.bin"
        out_path = os.path.join(out_dir, filename)

        # Stream to file to avoid loading whole file in memory
        with open(out_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
    except Exception as e:
        print(f"!!! File download failed for {file_id}: {e}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download PUBLIC Google Sheets/Drive files without auth."
    )
    p.add_argument(
        "--target",
        action="append",
        default=[],
        help="File ID or share URL (repeatable).",
    )
    p.add_argument("--file-list", help="Path to text file listing IDs/URLs.")
    p.add_argument(
        "-o", "--output", default=DEFAULT_OUTPUT_DIR, help="Output directory."
    )
    p.add_argument(
        "--sheet", action="store_true", help="Force treat raw ID as Sheet."
    )
    p.add_argument("--gid", help="Worksheet gid (for raw ID Sheets).")
    return p


def main():
    args = build_parser().parse_args()
    out_dir = os.path.abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)

    targets = collect_targets(args.target, args.file_list)
    if not targets:
        print("No targets provided.")
        return

    print(f"Resolved {len(targets)} target(s). Output: {out_dir}")
    for entry in targets:
        file_id, gid, is_sheet = parse_target(entry)
        if args.sheet and not is_sheet:  # override forced sheet
            is_sheet = True
            gid = args.gid or gid
        if is_sheet:
            download_sheet(file_id, gid, out_dir)
        else:
            download_drive_file(file_id, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
