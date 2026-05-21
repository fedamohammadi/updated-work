"""
SQL with Python and SQLAlchemy:
- sqlite3 basics: raw connections, cursors, and SQL statements
- SQLAlchemy engine: connecting to databases the idiomatic way
- Schema definition: MetaData, Table, Column, and data types
- CRUD operations: insert, select, update, and delete with Core API
- Reading SQL into pandas: read_sql_query and DataFrame integration
- Aggregations and joins: GROUP BY and multi-table queries
- Practical example: a two-table e-commerce schema with analysis
"""

import sqlite3
import numpy as np
import pandas as pd
from sqlalchemy import (
    create_engine, text, MetaData, Table, Column,
    Integer, String, Float, ForeignKey, func,
)
from sqlalchemy import insert as sa_insert, select as sa_select
from sqlalchemy import update as sa_update, delete as sa_delete


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# Shared sample data
# ==============================================================
# Five products with price and stock, and 40 randomised orders
# linking product IDs to quantities and purchase dates. Both
# tables are recreated fresh inside each demo function so every
# section is self-contained and can run in isolation.

PRODUCTS = [
    (1, "Laptop",      1199.99, 30),
    (2, "Headphones",    89.99, 120),
    (3, "Keyboard",      49.99,  80),
    (4, "Monitor",      349.99,  25),
    (5, "Webcam",        59.99,  60),
]


def _product_dicts() -> list:
    return [{"id": id_, "name": n, "price": p, "stock": s}
            for id_, n, p, s in PRODUCTS]


def _make_orders() -> list:
    np.random.seed(14)
    dates = pd.date_range("2024-01-01", periods=90).strftime("%Y-%m-%d").tolist()
    return [
        {"id": i + 1,
         "product_id": int(np.random.randint(1, 6)),
         "quantity":   int(np.random.randint(1, 6)),
         "date":       dates[int(np.random.randint(0, 90))]}
        for i in range(40)
    ]


def _build_schema():
    meta = MetaData()
    products = Table("products", meta,
        Column("id",    Integer, primary_key=True),
        Column("name",  String(100), nullable=False),
        Column("price", Float,       nullable=False),
        Column("stock", Integer,     nullable=False),
    )
    orders = Table("orders", meta,
        Column("id",         Integer, primary_key=True),
        Column("product_id", Integer, ForeignKey("products.id"), nullable=False),
        Column("quantity",   Integer, nullable=False),
        Column("date",       String(10), nullable=False),
    )
    return meta, products, orders


def _populated_engine():
    """Return (engine, products_table, orders_table) with all rows inserted."""
    meta, products, orders = _build_schema()
    engine = create_engine("sqlite:///:memory:", echo=False)
    meta.create_all(engine)
    with engine.connect() as conn:
        conn.execute(sa_insert(products), _product_dicts())
        conn.execute(sa_insert(orders),   _make_orders())
        conn.commit()
    return engine, products, orders
