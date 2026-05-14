import requests
import os
import logging
import tarfile
import io
from xml.etree import ElementTree as ET

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

UNPAYWALL_EMAIL = "pilarbourg@icloud.com"


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
        response = requests.get(url, timeout=30)
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
        response = requests.get(url, timeout=30)
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


def _download_tgz(url: str) -> bytes | None:
    """
    Downloads a tar.gz file from NCBI via HTTPS, converting legacy FTP URLs
    to the new deprecated HTTPS path.

    Parameters
    ----------
    url : str
        Original FTP or HTTPS URL to the tar.gz file.

    Returns
    -------
    bytes or None
        Raw file bytes, or None if the download failed.
    """
    try:
        https_url = url.replace(
            "ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/",
            "https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/"
        )
        logging.info(f"Downloading tarball from {https_url}")
        response = requests.get(https_url, timeout=60)
        if response.status_code == 200:
            return response.content
        logging.error(f"Failed to download tarball: status {response.status_code}")
        return None
    except Exception as e:
        logging.error(f"Error downloading tarball: {e}")
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
    save_path = f"downloads/PMC{pmcid}.pdf"

    if os.path.exists(save_path):
        logging.info(f"PMC{pmcid} already downloaded, skipping.")
        return True

    download_url, fmt = get_download_info(pmcid)

    if download_url:
        logging.info(f"Downloading PMC{pmcid} as {fmt} from PMC OA API")

        if fmt == "pdf":
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(download_url, headers=headers, timeout=60, allow_redirects=True)
                if response.status_code == 200:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    with open(save_path, "wb") as f:
                        f.write(response.content)
                    logging.info(f"Saved to {save_path}")
                    return True
                logging.error(f"Failed to download PDF: status {response.status_code}")
            except Exception as e:
                logging.error(f"Error downloading PDF for PMC{pmcid}: {e}")

        elif fmt == "tgz":
            content = _download_tgz(download_url)
            if content:
                return _save_pdf_from_tgz(content, save_path)

    if doi:
        logging.info(f"PMC OA API failed, trying Unpaywall for DOI {doi}")
        pdf_url = get_pdf_url_from_doi(doi)
        if pdf_url:
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(pdf_url, headers=headers, timeout=60, allow_redirects=True)
                if response.status_code == 200 and "pdf" in response.headers.get("Content-Type", "").lower():
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    with open(save_path, "wb") as f:
                        f.write(response.content)
                    logging.info(f"Saved to {save_path} via Unpaywall")
                    return True
                logging.error(f"Unpaywall download failed: status {response.status_code}")
            except Exception as e:
                logging.error(f"Error downloading via Unpaywall for PMC{pmcid}: {e}")

    logging.error(f"All download methods failed for PMC{pmcid}")
    return False
