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


# ==============================================================
# 1. sqlite3 Basics
# ==============================================================
# sqlite3 is Python's built-in interface to SQLite databases.
# A Connection holds the database; a Cursor runs SQL statements.
# executemany() sends one parameterised statement with many rows
# in a single call — faster than looping over execute().
# The ? placeholder prevents SQL injection by binding values
# separately rather than formatting them into the SQL string.

def demo_sqlite3_basics() -> None:
    conn = sqlite3.connect(":memory:")
    cur  = conn.cursor()

    cur.execute("""
        CREATE TABLE products (
            id    INTEGER PRIMARY KEY,
            name  TEXT    NOT NULL,
            price REAL    NOT NULL,
            stock INTEGER NOT NULL
        )
    """)
    cur.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", PRODUCTS)
    conn.commit()

    cur.execute("SELECT name, price, stock FROM products ORDER BY price DESC")
    rows = cur.fetchall()

    print(f"\n  {'Product':<14} | {'Price':>9} | {'Stock':>5}")
    print(f"  {'-'*14}-+-{'-'*9}-+-{'-'*5}")
    for name, price, stock in rows:
        print(f"  {name:<14} | ${price:>8.2f} | {stock:>5}")

    cur.execute("SELECT COUNT(*), AVG(price), SUM(stock) FROM products")
    cnt, avg_p, total_stock = cur.fetchone()
    print(f"\n  Count: {cnt}  |  Avg price: ${avg_p:.2f}  |  Total stock: {int(total_stock)}")
    conn.close()


# ==============================================================
# 2. SQLAlchemy Engine
# ==============================================================
# create_engine() returns an Engine that manages a connection pool.
# The URL format is "dialect+driver://user:pass@host/dbname".
# "sqlite:///:memory:" opens an in-process SQLite database.
# engine.connect() yields a Connection context manager; text()
# wraps a raw SQL string so :param placeholders can be bound
# safely without string formatting or SQL injection risk.

def demo_engine() -> None:
    engine = create_engine("sqlite:///:memory:", echo=False)

    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE products (
                id INTEGER PRIMARY KEY, name TEXT,
                price REAL, stock INTEGER
            )
        """))
        conn.execute(
            text("INSERT INTO products VALUES (:id, :name, :price, :stock)"),
            _product_dicts(),
        )
        conn.commit()

        result = conn.execute(
            text("SELECT name, price FROM products WHERE price > :lo ORDER BY price DESC"),
            {"lo": 100},
        )
        rows = result.fetchall()

    print(f"\n  Dialect: {engine.dialect.name}  |  Driver: {engine.dialect.driver}")
    print(f"\n  Products priced above $100:")
    for row in rows:
        print(f"    {row.name:<14}  ${row.price:.2f}")


# ==============================================================
# 3. Schema Definition: MetaData and Table
# ==============================================================
# MetaData is a registry that maps table names to Table objects.
# Column types (Integer, String, Float) are database-agnostic;
# SQLAlchemy translates them to the dialect's native type on DDL.
# ForeignKey("products.id") declares a referential integrity link.
# metadata.create_all(engine) issues CREATE TABLE for every
# registered table that does not yet exist in the target database.

def demo_schema_definition() -> None:
    meta, products, orders = _build_schema()
    engine = create_engine("sqlite:///:memory:", echo=False)
    meta.create_all(engine)

    print(f"\n  Registered tables: {list(meta.tables.keys())}")
    print(f"\n  products columns:")
    for col in products.columns:
        print(f"    {col.name:<14} {str(col.type):<16}  pk={col.primary_key}  nullable={col.nullable}")
    print(f"\n  orders columns:")
    for col in orders.columns:
        fk = list(col.foreign_keys)
        fk_str = f"  -> {next(iter(fk)).target_fullname}" if fk else ""
        print(f"    {col.name:<14} {str(col.type):<16}  pk={col.primary_key}{fk_str}")


# ==============================================================
# 4. CRUD Operations with SQLAlchemy Core
# ==============================================================
# The Core expression API builds SQL as Python objects, not strings.
# sa_select(table).where(table.c.col == val) uses Python operators
# to build a WHERE clause. sa_update().values(col=new_val) generates
# the SET clause. sa_delete().where() removes matching rows.
# All statements execute inside a connection; commit() persists changes.

def demo_crud() -> None:
    meta, products, orders = _build_schema()
    engine = create_engine("sqlite:///:memory:", echo=False)
    meta.create_all(engine)

    with engine.connect() as conn:
        conn.execute(sa_insert(products), _product_dicts())
        conn.commit()

        cheap = conn.execute(
            sa_select(products).where(products.c.price < 100)
        ).fetchall()
        print(f"\n  Products under $100:")
        for r in cheap:
            print(f"    {r.name:<14}  ${r.price:.2f}")

        conn.execute(
            sa_update(products)
            .where(products.c.name == "Webcam")
            .values(price=54.99, stock=75)
        )
        conn.commit()
        webcam = conn.execute(
            sa_select(products.c.price, products.c.stock)
            .where(products.c.name == "Webcam")
        ).fetchone()
        print(f"\n  Webcam after update: price=${webcam.price:.2f}  stock={webcam.stock}")

        conn.execute(sa_delete(products).where(products.c.id == 3))
        conn.commit()
        remaining = conn.execute(
            sa_select(func.count()).select_from(products)
        ).scalar()
        print(f"\n  Rows remaining after deleting id=3: {remaining}")


# ==============================================================
# 5. Reading SQL into pandas
# ==============================================================
# pd.read_sql_query(sql, con) executes a SQL string and returns a
# DataFrame whose column names match the SELECT list (or AS aliases).
# The con argument accepts a SQLAlchemy engine or connection.
# This workflow is powerful: use SQL for joining and filtering where
# the database is fast, then pandas for reshaping and analysis.

def demo_read_sql() -> None:
    engine, products, orders = _populated_engine()

    df_products = pd.read_sql_query("SELECT * FROM products ORDER BY price DESC", engine)
    df_orders   = pd.read_sql_query("SELECT * FROM orders", engine)

    print(f"\n  Products ({df_products.shape[0]} rows x {df_products.shape[1]} cols):")
    print(df_products.to_string(index=False))

    print(f"\n  Orders shape: {df_orders.shape}")
    print(f"\n  Units ordered per product:")
    unit_sum = (df_orders.groupby("product_id")["quantity"]
                          .sum()
                          .sort_values(ascending=False)
                          .reset_index())
    merged = unit_sum.merge(df_products[["id", "name"]], left_on="product_id", right_on="id")
    for _, row in merged.iterrows():
        print(f"    {row['name']:<14}  {int(row['quantity'])} units")


# ==============================================================
# 6. Aggregations and Joins
# ==============================================================
# GROUP BY collapses rows that share a key column and applies
# aggregate functions (SUM, AVG, COUNT) to the rest of the columns.
# JOIN combines rows from orders and products where product_id = id.
# Writing the aggregation in SQL is far more efficient than loading
# both tables into Python and merging when the data is large.

def demo_aggregations_joins() -> None:
    engine, _, _ = _populated_engine()

    sql = """
        SELECT  p.name,
                COUNT(o.id)                         AS num_orders,
                SUM(o.quantity)                     AS total_qty,
                ROUND(AVG(o.quantity), 2)           AS avg_qty,
                ROUND(SUM(o.quantity * p.price), 2) AS revenue
        FROM    orders   o
        JOIN    products p ON p.id = o.product_id
        GROUP BY p.name
        ORDER BY revenue DESC
    """
    df    = pd.read_sql_query(sql, engine)
    total = df["revenue"].sum()

    print(f"\n  {'Product':<14} | {'Orders':>6} | {'Qty':>5} | {'Avg':>5} | {'Revenue':>10} | Share")
    print(f"  {'-'*14}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*10}-+------")
    for _, row in df.iterrows():
        share = row["revenue"] / total * 100
        print(f"  {row['name']:<14} | {int(row['num_orders']):>6} | {int(row['total_qty']):>5} | "
              f"{row['avg_qty']:>5.1f} | ${row['revenue']:>9,.2f} | {share:.1f}%")
    print(f"  {'Total':<14}                                   ${total:>9,.2f}")
