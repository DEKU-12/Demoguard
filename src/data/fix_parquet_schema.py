
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

POLLUTED_ROOT = Path("outputs/datasets/pusht_polluted")
FIXED_COLS = ("observation.state", "action")
LIST_SIZE = 2


def _to_fixed_list(col: pa.ChunkedArray, size: int) -> pa.Array:
    """Flatten a list<float> column and rebuild as fixed_size_list<float32>[size]."""
    flat = col.combine_chunks().flatten().cast(pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, size)


def fix_file(pqf: str) -> None:
    table = pq.read_table(pqf)
    new_cols, new_fields = [], []
    for field in table.schema:
        if field.name in FIXED_COLS:
            arr = _to_fixed_list(table[field.name], LIST_SIZE)
            new_cols.append(arr)
            new_fields.append(pa.field(field.name, pa.list_(pa.float32(), LIST_SIZE)))
        else:
            new_cols.append(table[field.name].combine_chunks())
            new_fields.append(field)
    fixed = pa.Table.from_arrays(new_cols, schema=pa.schema(new_fields))
    pq.write_table(fixed, pqf)
    print(f"  fixed {pqf}")


if __name__ == "__main__":
    parquets = sorted(glob.glob(str(POLLUTED_ROOT / "data" / "**" / "*.parquet"),
                                recursive=True))
    if not parquets:
        raise FileNotFoundError(f"No parquet under {POLLUTED_ROOT}/data")
    for pqf in parquets:
        fix_file(pqf)
    print("\nDone. Re-checking schema:")
    sch = pq.read_schema(parquets[0])
    for c in FIXED_COLS:
        print(f"  {c}: {sch.field(c).type}")