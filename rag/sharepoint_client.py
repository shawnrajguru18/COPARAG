"""
SharePoint ingestion via Microsoft Graph API.

Required environment variables:
    AZURE_TENANT_ID       – Azure AD tenant ID
    AZURE_CLIENT_ID       – App registration client ID
    AZURE_CLIENT_SECRET   – App registration client secret
    SHAREPOINT_SITE_ID    – SharePoint site ID (or hostname+path, resolved below)
    SHAREPOINT_DRIVE_NAME – Document library name, e.g. "Shared Documents" (default)
    SHAREPOINT_FOLDER     – Folder path inside the library, e.g. "Accelya"

Getting the site/drive IDs the first time:
    python -m rag.sharepoint_client --list-sites
"""

from __future__ import annotations

import os
import io
import logging
from pathlib import Path
from typing import Generator

import requests

logger = logging.getLogger(__name__)

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".txt", ".xlsx"}


def _get_token() -> str:
    """Obtain a client-credentials access token from Azure AD."""
    tenant_id = os.environ["AZURE_TENANT_ID"]
    client_id = os.environ["AZURE_CLIENT_ID"]
    client_secret = os.environ["AZURE_CLIENT_SECRET"]
    url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    resp = requests.post(
        url,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "https://graph.microsoft.com/.default",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


class SharePointClient:
    """
    Lists and downloads files from a SharePoint document library folder.

    Usage
    -----
    client = SharePointClient()
    for file_meta in client.list_files():
        content = client.download(file_meta["download_url"])
        # content is bytes
    """

    def __init__(self) -> None:
        self.token = _get_token()
        self.site_id = os.environ["SHAREPOINT_SITE_ID"]
        self.drive_name = os.environ.get("SHAREPOINT_DRIVE_NAME", "Shared Documents")
        self.folder_path = os.environ.get("SHAREPOINT_FOLDER", "")
        self._drive_id: str | None = None

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}

    def _drive_id_for_library(self) -> str:
        if self._drive_id:
            return self._drive_id
        url = f"{GRAPH_BASE}/sites/{self.site_id}/drives"
        resp = requests.get(url, headers=self._headers(), timeout=30)
        resp.raise_for_status()
        drives = resp.json().get("value", [])
        for drive in drives:
            if drive.get("name", "").lower() == self.drive_name.lower():
                self._drive_id = drive["id"]
                return self._drive_id
        raise ValueError(
            f"Drive '{self.drive_name}' not found. Available: "
            + ", ".join(d.get("name", "") for d in drives)
        )

    def _list_folder_items(self, drive_id: str, folder: str) -> list[dict]:
        """Return all items recursively under the given folder path."""
        if folder:
            url = f"{GRAPH_BASE}/drives/{drive_id}/root:/{folder}:/children"
        else:
            url = f"{GRAPH_BASE}/drives/{drive_id}/root/children"

        items: list[dict] = []
        while url:
            resp = requests.get(url, headers=self._headers(), timeout=30)
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("value", []):
                if "folder" in item:
                    # Recurse into sub-folder
                    sub_path = f"{folder}/{item['name']}" if folder else item["name"]
                    items.extend(self._list_folder_items(drive_id, sub_path))
                elif "file" in item:
                    ext = Path(item["name"]).suffix.lower()
                    if ext in SUPPORTED_EXTENSIONS:
                        items.append(
                            {
                                "name": item["name"],
                                "path": f"{folder}/{item['name']}" if folder else item["name"],
                                "size": item.get("size", 0),
                                "last_modified": item.get("lastModifiedDateTime", ""),
                                "download_url": item.get(
                                    "@microsoft.graph.downloadUrl",
                                    item.get("@content.downloadUrl", ""),
                                ),
                                "item_id": item["id"],
                            }
                        )
            url = data.get("@odata.nextLink")
        return items

    def list_files(self) -> list[dict]:
        """Return metadata for all supported files in the configured folder."""
        drive_id = self._drive_id_for_library()
        files = self._list_folder_items(drive_id, self.folder_path)
        logger.info("Found %d files in SharePoint folder '%s'", len(files), self.folder_path)
        return files

    def download(self, download_url: str) -> bytes:
        """Download a file by its pre-authenticated Graph download URL."""
        resp = requests.get(download_url, timeout=120)
        resp.raise_for_status()
        return resp.content

    def download_to_file(self, file_meta: dict, dest_dir: str | Path) -> Path:
        """Download a file to *dest_dir* and return the local path."""
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / file_meta["name"]
        content = self.download(file_meta["download_url"])
        dest.write_bytes(content)
        logger.info("Downloaded %s (%d bytes)", dest.name, len(content))
        return dest


# ── CLI helper ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, json

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="SharePoint Graph API helper")
    parser.add_argument("--list-sites", action="store_true", help="List accessible SharePoint sites")
    parser.add_argument("--list-files", action="store_true", help="List files in configured folder")
    args = parser.parse_args()

    token = _get_token()
    headers = {"Authorization": f"Bearer {token}"}

    if args.list_sites:
        resp = requests.get(f"{GRAPH_BASE}/sites?search=*", headers=headers, timeout=30)
        resp.raise_for_status()
        for site in resp.json().get("value", []):
            print(f"{site['id']!r:60s}  {site.get('displayName', '')}")
    elif args.list_files:
        client = SharePointClient()
        files = client.list_files()
        print(json.dumps(files, indent=2))
    else:
        parser.print_help()
