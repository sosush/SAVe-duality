import os
from pyiceberg.catalog import load_catalog
from pyiceberg.schema import Schema
from pyiceberg.types import StringType, FloatType, NestedField

os.makedirs("data/iceberg_warehouse", exist_ok=True)

# Zero-cost local SQLite catalog with Parquet file warehouse
catalog = load_catalog(
    "local_lakehouse",
    **{
        "type": "sql",
        "uri": "sqlite:///data/iceberg_catalog.db",
        "warehouse": "file://" + os.path.abspath("data/iceberg_warehouse"),
    }
)

schema = Schema(
    NestedField(field_id=1, name="timestamp", field_type=StringType(), required=True),
    NestedField(field_id=2, name="incident", field_type=StringType(), required=True),
    NestedField(field_id=3, name="motion", field_type=FloatType(), required=True),
    NestedField(field_id=4, name="audio", field_type=FloatType(), required=True)
)

def init_iceberg():
    catalog.create_namespace_if_not_exists("smart_security")
    table_id = "smart_security.cubicle_incidents"
    try:
        table = catalog.create_table(table_id, schema=schema)
        print(f"[✓] Created Apache Iceberg Table: {table_id}")
    except Exception:
        table = catalog.load_table(table_id)
        print(f"[✓] Loaded existing Apache Iceberg Table: {table_id}")
    return table

if __name__ == "__main__":
    init_iceberg()