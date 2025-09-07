import re
import time
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from readability import Document

DEFAULT_HEADERS = {
    "User-Agent": (
        # A realistic desktop UA helps avoid basic bot blocks
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
}

def _readability_extract(html: str) -> tuple[str, str]:
    """
    Use readability to pull out the main content and title.
    Returns (title, text)
    """
    doc = Document(html)
    title = (doc.short_title() or "").strip()
    summary_html = doc.summary() or ""
    soup = BeautifulSoup(summary_html, "lxml")
    # Normalize whitespace and keep paragraphs separated
    text = "\n".join(
        [re.sub(r"\s+", " ", p.get_text(" ", strip=True)).strip()
         for p in soup.find_all(["p", "li", "h2", "h3", "h4"])]
    )
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return title, text

def _fallback_title(html: str, url: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    # Prefer og:title if available
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og["content"].strip()
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return url

def _jina_reader_url(url: str) -> str:
    """
    Build a r.jina.ai reader URL. It expects http:// scheme in the path after the host.
    We strip the original scheme and always prefix http:// for the reader.
    """
    parsed = urlparse(url)
    bare = f"{parsed.netloc}{parsed.path}"
    if parsed.query:
        bare += f"?{parsed.query}"
    return f"https://r.jina.ai/http://{bare}"

def fetch_website_content(url: str, timeout: int = 25) -> dict:
    """
    Fetch a web page, extract readable text.
    If blocked or content too short, fallback to r.jina.ai reader.
    Returns: { 'title': str, 'content': str }
             or { 'title': url, 'content': '__FETCH_ERROR__: <reason>' }
    """
    # --- Attempt 1: direct fetch with realistic headers ---
    try:
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout, allow_redirects=True)
        status = resp.status_code
        html = resp.text or ""
        if status == 200 and html.strip():
            title = _fallback_title(html, url)
            art_title, text = _readability_extract(html)
            if art_title:
                title = art_title
            # If the readable text looks too short, try fallback
            if text and len(text) > 300:
                return {"title": title, "content": text}
        else:
            # 403/404/empty → try fallback
            pass
    except Exception as e:
        # network error → try fallback
        last_err = f"direct fetch error: {e}"

    # --- Attempt 2: r.jina.ai fallback (works on many blocked/JS pages) ---
    try:
        proxy_url = _jina_reader_url(url)
        # small backoff helps sometimes
        time.sleep(0.25)
        prox = requests.get(proxy_url, headers=DEFAULT_HEADERS, timeout=timeout)
        if prox.status_code == 200 and prox.text.strip():
            text = prox.text.strip()
            # r.jina.ai returns plain text; create a decent title
            title = url
            # If it's too short, treat as failure
            if len(text) > 300:
                return {"title": title, "content": text}
            else:
                return {"title": title, "content": "__FETCH_ERROR__: content too short from reader"}
        else:
            return {
                "title": url,
                "content": f"__FETCH_ERROR__: reader status {prox.status_code}"
            }
    except Exception as e2:
        return {
            "title": url,
            "content": f"__FETCH_ERROR__: fallback reader error: {e2}"
        }
