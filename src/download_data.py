"""Download and verify the exact Kaggle snapshot used in the analysis."""

# Import only standard-library tools needed for the fixed download.
import hashlib
import io
from pathlib import Path
import urllib.request
import zipfile

# Pin both the Kaggle version and the extracted CSV contents.
ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.kaggle.com/api/v1/datasets/download/kanchana1990/llm-price-performance-tracker-march-2026?datasetVersionNumber=1"
FILENAME = "llm_price_performance_tracker_2026-03-31.csv"
CSV_SHA256 = "1c83338dc40f826ce19f3c6d17a3e03970979587fb860dccc79846f691e07177"
ZIP_SHA256 = "d8984829ce0fa195ddfac5d2ab7897cd71fe5a1506aa1be31614eb7c8c4fee8e"


# Check a cached file or fetch the pinned archive without a Kaggle account.
def download():
    destination = ROOT / "data" / "raw" / FILENAME
    if destination.exists():
        csv_bytes = destination.read_bytes()
    else:
        request = urllib.request.Request(URL, headers={"User-Agent": "llm-api-pricing-analysis/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            archive_bytes = response.read()
        if hashlib.sha256(archive_bytes).hexdigest() != ZIP_SHA256:
            raise ValueError("The pinned ZIP checksum changed; inspect the source before continuing.")
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            csv_bytes = archive.read(FILENAME)

    # Refuse changed input instead of silently running on a different dataset.
    checksum = hashlib.sha256(csv_bytes).hexdigest()
    if checksum != CSV_SHA256:
        raise ValueError("CSV checksum mismatch; expected the documented March 31 version 1 snapshot.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(csv_bytes)
    print(f"Verified Kaggle v1: {FILENAME} ({len(csv_bytes):,} bytes)")
    print(f"SHA-256: {checksum}")
    return destination


# Allow the same verified download from the terminal and the notebook.
if __name__ == "__main__":
    download()
