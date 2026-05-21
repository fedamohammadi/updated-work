"""
Web Data: Requests, APIs, and JSON:
- HTTP basics: GET requests, status codes, and response objects
- JSON: parsing response bodies and serialising Python objects
- The requests library: cleaner interface to HTTP
- Query parameters and headers: filtering and authenticating API calls
- Error handling: status codes, timeouts, and network exceptions
- Transforming API data: flattening nested JSON into a DataFrame
- Practical example: fetching and analysing public REST API data
"""

import json
import urllib.request
import urllib.parse
import urllib.error
import numpy as np
import pandas as pd

BASE_URL = "https://jsonplaceholder.typicode.com"


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# Shared helpers and fallback data
# ==============================================================
# fetch_json() wraps urllib.request with a timeout and falls back
# to mock_data when the network is unavailable. This keeps every
# demo runnable offline while showing the patterns used against a
# real API. MOCK_POSTS and MOCK_USERS mirror the JSONPlaceholder
# response shape exactly so the transformation code is identical.

MOCK_POSTS = [
    {"userId": 1, "id": 1, "title": "Python data analysis fundamentals",
     "body": "NumPy arrays enable fast vectorised operations over large datasets."},
    {"userId": 1, "id": 2, "title": "Pandas DataFrames explained",
     "body": "A DataFrame is a two-dimensional labelled data structure with typed columns."},
    {"userId": 2, "id": 3, "title": "Visualising data with matplotlib",
     "body": "The Figure and Axes API separates the canvas from the individual plot area."},
    {"userId": 2, "id": 4, "title": "Seaborn statistical graphics",
     "body": "Seaborn builds on matplotlib and adds statistical defaults and a DataFrame API."},
    {"userId": 3, "id": 5, "title": "SQL and pandas integration",
     "body": "read_sql_query bridges a relational database and a DataFrame in one call."},
    {"userId": 3, "id": 6, "title": "Working with REST APIs in Python",
     "body": "The requests library simplifies HTTP with a clean and intuitive interface."},
    {"userId": 4, "id": 7, "title": "Cleaning missing data in pandas",
     "body": "dropna and fillna handle NaN values using different imputation strategies."},
    {"userId": 4, "id": 8, "title": "GroupBy and aggregation patterns",
     "body": "groupby().agg() applies multiple functions to each group simultaneously."},
    {"userId": 5, "id": 9, "title": "Reproducible reports with Quarto",
     "body": "Quarto renders .qmd files to HTML, PDF, and Word documents automatically."},
    {"userId": 5, "id": 10, "title": "Time series analysis with pandas",
     "body": "DatetimeIndex enables resampling and rolling window computations."},
]

MOCK_USERS = [
    {"id": 1, "name": "Alice",  "username": "alice",  "email": "alice@example.com",
     "company": {"name": "DataCo"},  "address": {"city": "Austin"}},
    {"id": 2, "name": "Bob",    "username": "bob",    "email": "bob@example.com",
     "company": {"name": "MLHub"},   "address": {"city": "Seattle"}},
    {"id": 3, "name": "Carol",  "username": "carol",  "email": "carol@example.com",
     "company": {"name": "DataCo"},  "address": {"city": "Boston"}},
    {"id": 4, "name": "David",  "username": "david",  "email": "david@example.com",
     "company": {"name": "WebTech"}, "address": {"city": "Denver"}},
    {"id": 5, "name": "Eve",    "username": "eve",    "email": "eve@example.com",
     "company": {"name": "MLHub"},   "address": {"city": "Portland"}},
]


def fetch_json(url: str, fallback) -> object:
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return fallback


# ==============================================================
# 1. HTTP Basics: GET Requests
# ==============================================================
# urllib.request.urlopen() opens a URL and returns an HTTP response.
# resp.status is the numeric status code: 200 = OK, 404 = not found.
# resp.read() returns the raw bytes of the body; .decode() makes it
# a string. The Accept header tells the server what content type to
# return. urllib is part of the standard library — no installation needed.

def demo_http_basics() -> None:
    url = f"{BASE_URL}/posts/1"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            status       = resp.status
            content_type = resp.headers.get("Content-Type", "n/a")
            post         = json.loads(resp.read().decode())
        live = True
    except Exception:
        status, content_type, post, live = 0, "application/json", MOCK_POSTS[0], False

    print(f"\n  URL          : {url}")
    print(f"  Status       : {status if live else '(offline -- mock data)'}")
    print(f"  Content-Type : {content_type}")
    print(f"\n  Response body (parsed):")
    for k, v in post.items():
        print(f"    {k:<10}: {str(v)[:60]}")


# ==============================================================
# 2. JSON: Parsing and Serialising
# ==============================================================
# json.loads() decodes a JSON string into Python objects: objects
# become dicts, arrays become lists, true/false become bool.
# json.dumps() does the reverse. indent= pretty-prints the output.
# ensure_ascii=False preserves non-ASCII characters (accented letters,
# CJK). sort_keys=True produces consistent output across Python runs.

def demo_json() -> None:
    raw = """{
        "user":   {"id": 7, "name": "Alice", "active": true},
        "scores": [95, 88, 72, 100],
        "meta":   {"pages": 3, "total": 4}
    }"""
    data = json.loads(raw)

    print(f"\n  Parsed types from JSON:")
    print(f"    data['user']   -> {type(data['user']).__name__}: {data['user']}")
    print(f"    data['scores'] -> {type(data['scores']).__name__}: {data['scores']}")
    print(f"    data['meta']   -> {type(data['meta']).__name__}: {data['meta']}")

    data["scores"].append(91)
    data["user"]["active"] = False
    serialised = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True)

    print(f"\n  Re-serialised with indent=2, sort_keys=True:")
    for line in serialised.splitlines():
        print(f"  {line}")

    avg = sum(data["scores"]) / len(data["scores"])
    print(f"\n  Avg score: {avg:.1f}  |  User active: {data['user']['active']}")


# ==============================================================
# 3. The requests Library
# ==============================================================
# requests.get() wraps urllib with a cleaner API. Response.json()
# auto-parses the body without a separate json.loads() call.
# params= appends query parameters to the URL with proper encoding.
# Session() reuses a TCP connection pool and lets you set default
# headers once, reducing overhead across many requests to one host.

def demo_requests_pattern() -> None:
    url   = f"{BASE_URL}/posts"
    posts = fetch_json(f"{url}?userId=1&_limit=5", MOCK_POSTS[:5])

    print(f"\n  With urllib (used in this file for portability):")
    print(f"    req = urllib.request.Request(url, headers={{...}})")
    print(f"    with urllib.request.urlopen(req, timeout=5) as resp:")
    print(f"        data = json.loads(resp.read().decode())")

    print(f"\n  Equivalent with requests (simpler API):")
    print(f"    import requests")
    print(f"    r = requests.get('{url}', params={{'userId': 1, '_limit': 5}})")
    print(f"    posts = r.json()         # auto-parses JSON body")
    print(f"    print(r.status_code)     # 200")
    print(f"    print(r.elapsed)         # response time")

    print(f"\n  Posts for userId=1 (first 5):")
    for p in posts[:5]:
        print(f"    id={p['id']}  {p['title'][:52]}")


# ==============================================================
# 4. Query Parameters and Headers
# ==============================================================
# Query parameters filter or page API responses — they appear after
# ? in the URL as key=value pairs joined by &. urllib.parse.urlencode()
# builds this string safely from a dict (handles special characters).
# Headers carry metadata: Accept, Authorization (Bearer tokens), and
# Content-Type. Never hard-code secrets — read from environment vars.

def demo_params_headers() -> None:
    import os

    params  = {"userId": 2, "_limit": 3, "_sort": "id", "_order": "asc"}
    encoded = urllib.parse.urlencode(params)
    url     = f"{BASE_URL}/posts?{encoded}"
    posts   = fetch_json(url, [p for p in MOCK_POSTS if p["userId"] == 2][:3])

    print(f"\n  URL with encoded params:")
    print(f"  {url}")

    print(f"\n  Common request headers:")
    print(f"    Accept        : application/json")
    print(f"    Authorization : Bearer <token>   # os.getenv('API_TOKEN')")
    print(f"    Content-Type  : application/json # for POST / PUT bodies")
    print(f"    User-Agent    : MyApp/1.0")

    print(f"\n  Building headers in code:")
    print(f"    headers = {{'Accept': 'application/json',")
    print(f"               'Authorization': 'Bearer ' + os.getenv('API_TOKEN', '')}}")

    print(f"\n  Posts for userId=2 (limit 3):")
    for p in posts[:3]:
        print(f"    id={p['id']}  {p['title'][:52]}")
