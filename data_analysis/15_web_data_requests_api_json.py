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
