"""Throwaway diagnostic: run against a real product URL to see exactly what
Amazon returns, so we can tell whether it's a bot-block or a pattern mismatch.

Usage:
    python diagnose_scrape.py "https://www.amazon.com/dp/XXXXXXXXXX"
"""
import sys

import requests

url = sys.argv[1]
headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
resp = requests.get(url, headers=headers, timeout=10)
print("Final URL:", resp.url)
print("Status code:", resp.status_code)
print("Content length:", len(resp.text))
print("Looks like a captcha/bot wall:", "captcha" in resp.text.lower() or "opfcaptcha" in resp.text.lower())
print("Contains 'og:image':", "og:image" in resp.text)
print("Contains 'hiRes':", '"hiRes"' in resp.text)
print("Contains 'landingImage':", "landingImage" in resp.text)
print()
if "og:image" in resp.text:
    idx = resp.text.find("og:image")
    print("Context around og:image tag:")
    print(resp.text[max(0, idx - 100):idx + 200])
else:
    print("First 1000 chars of response:")
    print(resp.text[:1000])
