"""Compile the paper draft (tasks/paper_outline.md) to .docx and upload it to Google Drive as attn_fuse_draft_2026.docx.

The first run creates the Drive file; later runs replace its content in place, so the link stays the same. Styles
come from the user's original outline (tasks/AttentionFuse Outline OSDI 25'.docx) as pandoc's reference document.
Credentials (OAuth token with a refresh token, scope drive.file) live outside the repo in
~/.config/attn_fuse/gdrive_token.json; the Drive file id is cached next to them. The post-commit hook runs this script
whenever a commit touches the draft.
  python tasks/paper_sync_gdrive.py [--dry-run]
Claude, 2026-10-26.
"""
import argparse
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DRAFT = ROOT / "tasks" / "paper_outline.md"
REFERENCE_DOC = ROOT / "tasks" / "AttentionFuse Outline OSDI 25'.docx"
PANDOC = Path("/opt/tiger/tools/pypandoc/pypandoc/files/pandoc")
CONFIG = Path.home() / ".config" / "attn_fuse"
TOKEN = CONFIG / "gdrive_token.json"
FILE_ID = CONFIG / "gdrive_file_id"
OUTPUT = CONFIG / "attn_fuse_draft_2026.docx"
NAME = "attn_fuse_draft_2026.docx"
MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


BORDERS = ("<w:tblBorders>" + "".join(
    f'<w:{side} w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
    for side in ("top", "left", "bottom", "right", "insideH", "insideV")) + "</w:tblBorders>")
HEADER_SHADE = '<w:shd w:val="clear" w:color="auto" w:fill="E7E6E6"/>'


def format_tables(document):
    """Pandoc's tables use a "Table" style that the reference document does not define, so Word and Google Docs show
    them borderless, like text. Give every table grid borders at full page width, and shade and bold its header row."""
    document = document.replace('<w:tblW w:type="auto" w:w="0" />', '<w:tblW w:type="pct" w:w="5000" />')
    document = re.sub(r"<w:tblPr>(.*?)</w:tblPr>",
                      lambda m: "<w:tblPr>" + m.group(1).replace("<w:tblLook", BORDERS + "<w:tblLook", 1)
                      + "</w:tblPr>", document, flags=re.S)

    def header(row):
        row = row.replace("<w:tcPr />", f"<w:tcPr>{HEADER_SHADE}</w:tcPr>")
        return re.sub(r"<w:r>(<w:rPr>.*?</w:rPr>)?<w:t", lambda m: "<w:r><w:rPr><w:b/></w:rPr><w:t"
                      if not m.group(1) else "<w:r>" + m.group(1).replace("<w:rPr>", "<w:rPr><w:b/>", 1) + "<w:t",
                      row)
    return re.sub(r'<w:tr><w:trPr><w:tblHeader w:val="on" /></w:trPr>.*?</w:tr>', lambda m: header(m.group(0)),
                  document, flags=re.S)


def compile_docx():
    raw = OUTPUT.with_suffix(".raw.docx")
    command = [str(PANDOC), str(DRAFT), "-f", "gfm", "-t", "docx", "-o", str(raw), "--toc"]
    if REFERENCE_DOC.exists():
        command += ["--reference-doc", str(REFERENCE_DOC)]
    subprocess.run(command, check=True)
    with zipfile.ZipFile(raw) as source, zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "word/document.xml":
                data = format_tables(data.decode("utf-8")).encode("utf-8")
            target.writestr(item, data)
    raw.unlink()
    return OUTPUT


def table_report(path):
    document = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8")
    tables = re.findall(r"<w:tbl>.*?</w:tbl>", document, flags=re.S)
    shapes = [(len(re.findall(r"<w:tr>", table)), len(re.findall(r"<w:gridCol ", table))) for table in tables]
    bordered = sum("<w:tblBorders>" in table for table in tables)
    return len(tables), bordered, shapes


def access_token():
    token = json.loads(TOKEN.read_text())
    response = requests.post(token["token_uri"], timeout=60, data=dict(
        client_id=token["client_id"], client_secret=token["client_secret"],
        refresh_token=token["refresh_token"], grant_type="refresh_token"))
    response.raise_for_status()
    return response.json()["access_token"]


def find_existing(headers):
    if FILE_ID.exists():
        file_id = FILE_ID.read_text().strip()
        response = requests.get(f"https://www.googleapis.com/drive/v3/files/{file_id}", headers=headers, timeout=60,
                                params=dict(fields="id,trashed"))
        if response.ok and not response.json().get("trashed"):
            return file_id
    # drive.file scope: the search sees only files this client created.
    response = requests.get("https://www.googleapis.com/drive/v3/files", headers=headers, timeout=60,
                            params=dict(q=f"name = '{NAME}' and trashed = false", fields="files(id)"))
    response.raise_for_status()
    files = response.json()["files"]
    return files[0]["id"] if files else None


def upload(path):
    headers = {"Authorization": f"Bearer {access_token()}"}
    file_id = find_existing(headers)
    body = path.read_bytes()
    if file_id:
        response = requests.patch(f"https://www.googleapis.com/upload/drive/v3/files/{file_id}", timeout=300,
                                  headers={**headers, "Content-Type": MIME},
                                  params=dict(uploadType="media", fields="id,name,modifiedTime,webViewLink"),
                                  data=body)
    else:
        boundary = "attn_fuse_draft_boundary"
        metadata = json.dumps(dict(name=NAME, mimeType=MIME))
        multipart = (f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n{metadata}\r\n"
                     f"--{boundary}\r\nContent-Type: {MIME}\r\n\r\n").encode() + body + f"\r\n--{boundary}--".encode()
        response = requests.post("https://www.googleapis.com/upload/drive/v3/files", timeout=300,
                                 headers={**headers, "Content-Type": f"multipart/related; boundary={boundary}"},
                                 params=dict(uploadType="multipart", fields="id,name,modifiedTime,webViewLink"),
                                 data=multipart)
    response.raise_for_status()
    info = response.json()
    FILE_ID.write_text(info["id"] + "\n")
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="compile only")
    arguments = parser.parse_args()
    path = compile_docx()
    count, bordered, shapes = table_report(path)
    print(f"PAPER_DOCX {path} {path.stat().st_size} bytes; {count} tables ({bordered} bordered), rows x cols {shapes}",
          flush=True)
    expected = len(re.findall(r"^\|[\s:|-]+\|\s*$", DRAFT.read_text(), flags=re.M))
    if count != expected or bordered != count:
        raise SystemExit(f"table check failed: {expected} markdown tables, {count} in the docx, {bordered} bordered")
    if arguments.dry_run:
        return
    info = upload(path)
    print("PAPER_SYNCED", json.dumps(info), flush=True)


if __name__ == "__main__":
    sys.exit(main())
