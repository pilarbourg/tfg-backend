import requests
import os
import logging
import tarfile
import io
import re
from xml.etree import ElementTree as ET

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://pmc.ncbi.nlm.nih.gov/",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

def _https_candidates(url: str) -> list[str]:
    """
    Returns HTTPS candidate URLs for a PMC FTP URL, in priority order.

    requests cannot fetch ftp:// URLs (no connection adapter), but NCBI
    serves the same files over HTTPS on the same host. The exact path may
    or may not include a 'deprecated/' segment depending on the resource,
    so we return both the plain scheme-swap and the deprecated variant and
    let the caller try them in turn.

    Parameters
    ----------
    url : str
        Original URL (ftp:// or https://).

    Returns
    -------
    list[str]
        Ordered list of HTTPS URLs to attempt. A non-FTP URL is returned
        unchanged as a single-element list.
    """
    if not url.startswith("ftp://"):
        return [url]

    plain = url.replace("ftp://", "https://", 1)
    deprecated = plain.replace(
        "https://ftp.ncbi.nlm.nih.gov/pub/pmc/",
        "https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/",
        1,
    )
    # De-duplicate while preserving order (in case the replace was a no-op).
    candidates = []
    for c in (plain, deprecated):
        if c not in candidates:
            candidates.append(c)
    return candidates


def get_download_info(pmcid: str) -> tuple[str, str] | tuple[None, None]:
    """
    Retrieves the direct download URL and format for a given PMCID using the PMC OA API.

    Parameters
    ----------
    pmcid : str
        PubMed Central identifier for the paper.

    Returns
    -------
    tuple[str, str] or tuple[None, None]
        Tuple of (url, format) where format is 'pdf' or 'tgz', or (None, None) if unavailable.
    """
    url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC{pmcid}"
    try:
        response = SESSION.get(url, timeout=30)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        pdf_url = None
        tgz_url = None

        for link in root.iter("link"):
            fmt = link.attrib.get("format")
            href = link.attrib.get("href", "")
            if fmt == "pdf":
                pdf_url = href
            elif fmt == "tgz":
                tgz_url = href

        if pdf_url:
            return pdf_url, "pdf"
        elif tgz_url:
            return tgz_url, "tgz"

        logging.warning(f"No downloadable format found for PMC{pmcid}")
        return None, None

    except Exception as e:
        logging.error(f"Error fetching download info for PMC{pmcid}: {e}")
        return None, None


def get_pdf_url_from_doi(doi: str) -> str | None:
    """
    Retrieves the best open access PDF URL for a given DOI using the Unpaywall API.

    Parameters
    ----------
    doi : str
        Digital Object Identifier for the paper.

    Returns
    -------
    str or None
        Direct URL to the PDF, or None if unavailable.
    """
    try:
        url = f"https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}"
        response = SESSION.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        best = data.get("best_oa_location")
        if best and best.get("url_for_pdf"):
            return best["url_for_pdf"]

        for location in data.get("oa_locations", []):
            if location.get("url_for_pdf"):
                return location["url_for_pdf"]

        logging.warning(f"No PDF URL found in Unpaywall for DOI {doi}")
        return None

    except Exception as e:
        logging.error(f"Error fetching Unpaywall data for DOI {doi}: {e}")
        return None


def _save_pdf_response(content: bytes, save_path: str) -> bool:
    """Writes PDF bytes to save_path, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(content)
    return True


def _download_pdf_direct(url: str, save_path: str, require_pdf_content_type: bool = False) -> bool:
    """
    Downloads a PDF directly over HTTPS, trying FTP->HTTPS candidates in order.

    Parameters
    ----------
    url : str
        Source URL (ftp:// or https://).
    save_path : str
        Destination path for the PDF.
    require_pdf_content_type : bool
        If True, only accept responses whose Content-Type contains 'pdf'
        (used for Unpaywall, where landing pages are sometimes returned).

    Returns
    -------
    bool
        True if a PDF was saved, False otherwise.
    """
    for candidate in _https_candidates(url):
        try:
            logging.info(f"Downloading PDF from {candidate}")
            response = SESSION.get(candidate, headers=HEADERS, timeout=60, allow_redirects=True)
            if response.status_code != 200:
                logging.warning(f"PDF fetch returned status {response.status_code} for {candidate}")
                continue
            if require_pdf_content_type and "pdf" not in response.headers.get("Content-Type", "").lower():
                logging.warning(f"Response was not a PDF (Content-Type) for {candidate}")
                continue
            _save_pdf_response(response.content, save_path)
            logging.info(f"Saved to {save_path}")
            return True
        except Exception as e:
            logging.error(f"Error downloading PDF from {candidate}: {e}")
            continue
    return False


def _download_tgz(url: str) -> bytes | None:
    """
    Downloads a tar.gz file from NCBI via HTTPS, trying FTP->HTTPS candidates.

    Parameters
    ----------
    url : str
        Original FTP or HTTPS URL to the tar.gz file.

    Returns
    -------
    bytes or None
        Raw file bytes, or None if the download failed.
    """
    for candidate in _https_candidates(url):
        try:
            logging.info(f"Downloading tarball from {candidate}")
            response = SESSION.get(candidate, timeout=60)
            if response.status_code == 200:
                return response.content
            logging.warning(f"Tarball fetch returned status {response.status_code} for {candidate}")
        except Exception as e:
            logging.error(f"Error downloading tarball from {candidate}: {e}")
            continue
    logging.error(f"Failed to download tarball from any candidate for {url}")
    return None


def _save_pdf_from_tgz(content: bytes, save_path: str) -> bool:
    """
    Extracts and saves the first PDF found inside a tar.gz archive.

    Parameters
    ----------
    content : bytes
        Raw bytes of the tar.gz file.
    save_path : str
        Destination path for the extracted PDF.

    Returns
    -------
    bool
        True if a PDF was found and saved, False otherwise.
    """
    try:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".pdf"):
                    f = tar.extractfile(member)
                    if f:
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        with open(save_path, "wb") as out:
                            out.write(f.read())
                        logging.info(f"Extracted PDF from tarball, saved to {save_path}")
                        return True

        logging.error(f"No PDF found inside tarball for {save_path}")
        return False

    except Exception as e:
        logging.error(f"Error extracting tarball: {e}")
        return False


def download_pmc_pdf(pmcid: str, doi: str | None = None) -> bool:
    """
    Downloads a PubMed Central article PDF. Tries the PMC OA API first,
    then falls back to Unpaywall if a DOI is available.

    Parameters
    ----------
    pmcid : str
        PubMed Central identifier for the paper.
    doi : str or None
        Digital Object Identifier, used as fallback via Unpaywall.

    Returns
    -------
    bool
        True if the download succeeded, False otherwise.
    """
    if not pmcid:
        logging.error("No PMCID provided")
        return False

    save_path = f"downloads/PMC{pmcid}.pdf"

    if os.path.exists(save_path):
        logging.info(f"PMC{pmcid} already downloaded, skipping.")
        return True

    download_url, fmt = get_download_info(pmcid)

    if download_url:
        logging.info(f"Downloading PMC{pmcid} as {fmt} from PMC OA API")

        if fmt == "pdf":
            if _download_pdf_direct(download_url, save_path):
                return True

        elif fmt == "tgz":
            content = _download_tgz(download_url)
            if content:
                return _save_pdf_from_tgz(content, save_path)
            
    logging.info(f"Trying direct PMC PDF endpoint for PMC{pmcid}")

    if try_pmc_article_pdf(pmcid, save_path):
        logging.info(f"Saved PMC{pmcid} via direct PMC PDF endpoint")
        return True

    if doi:
        logging.info(f"PMC OA API failed, trying Unpaywall for DOI {doi}")
        pdf_url = get_pdf_url_from_doi(doi)
        if pdf_url:
            if _download_pdf_direct(pdf_url, save_path, require_pdf_content_type=True):
                logging.info(f"Saved {save_path} via Unpaywall")
                return True
            logging.error(f"Unpaywall download failed for PMC{pmcid}")

    logging.error(f"All download methods failed for PMC{pmcid}")
    return False

def try_pmc_article_pdf(pmcid: str, save_path: str) -> bool:
    url = (
        "https://europepmc.org/backend/ptpmcrender.fcgi"
        f"?accid=PMC{pmcid}&blobtype=pdf"
    )

    try:
        logging.info(f"Trying Europe PMC PDF URL: {url}")

        r = requests.get(
            url,
            timeout=60,
            stream=True,
        )

        logging.info(f"Status: {r.status_code}")
        logging.info(f"Content-Type: {r.headers.get('Content-Type')}")

        content = r.content

        if content.startswith(b"%PDF"):

            _save_pdf_response(content, save_path)

            logging.info(f"Saved PDF to {save_path}")

            return True

        logging.warning("Europe PMC response was not a valid PDF")
        logging.info(content[:200])

    except Exception as e:
        logging.error(f"Europe PMC PDF failed: {e}")

    return False