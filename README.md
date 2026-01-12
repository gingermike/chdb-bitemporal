# timeseries-chdb

Bitemporal timeseries processing using chdb (embedded ClickHouse). Handles timeline reconstruction for overlapping updates in a single SQL pass.

## What it does

Given a current state table and a batch of updates, computes:
- **expires**: existing records that need to be closed (set `as_of_to`)
- **inserts**: new records to insert with the reconstructed timeline

The algorithm handles:
- Multiple overlapping updates per ID in a single batch
- Later updates win in overlapping regions
- Adjacent segments with identical values are merged
- Composite ID columns (e.g., portfolio + security)
- Multiple value columns with change detection via hashing
- Two update modes: **delta** (overlay) and **full_state** (flush and fill with tombstoning)

## Installation

```bash
uv add timeseries-chdb
```

Or with pip:
```bash
pip install timeseries-chdb
```

## Usage

```python
from datetime import date
import pandas as pd
from bitemporal import compute_changes

# Current state - what's in your database
current_state = pd.DataFrame({
    "id": ["A"],
    "value": [100],
    "effective_from": [date(2024, 1, 1)],
    "effective_to": [date(2024, 12, 31)],
    "as_of_from": [date(2024, 1, 1)],
    "as_of_to": [None],  # Active record
})

# Incoming updates
updates = pd.DataFrame({
    "id": ["A"],
    "value": [200],
    "effective_from": [date(2024, 6, 1)],
    "effective_to": [date(2024, 8, 31)],
})

# Compute changes
result = compute_changes(
    current_state,
    updates,
    id_columns=["id"],
    value_columns=["value"],
    system_date=date(2024, 5, 15),
)

# result.expires - records to close with as_of_to = 2024-05-15
# result.inserts - new timeline segments to insert
```

The above produces:
- 1 expire (the original record)
- 3 inserts:
  - `[2024-01-01, 2024-06-01)` value=100
  - `[2024-06-01, 2024-08-31)` value=200
  - `[2024-08-31, 2024-12-31)` value=100

### Full State Mode (Flush and Fill)

Use `update_mode="full_state"` when updates represent the complete current state. IDs present in current state but missing from updates will be tombstoned (their `effective_to` is set to `system_date`).

```python
# Current state has IDs A and B
current_state = pd.DataFrame({
    "id": ["A", "B"],
    "value": [100, 200],
    "effective_from": pd.to_datetime(["2024-01-01", "2024-01-01"]),
    "effective_to": pd.to_datetime(["2024-12-31", "2024-12-31"]),
    "as_of_from": pd.to_datetime(["2024-01-01", "2024-01-01"]),
    "as_of_to": [None, None],
})

# Updates only contain A - B will be tombstoned
updates = pd.DataFrame({
    "id": ["A"],
    "value": [150],
    "effective_from": pd.to_datetime(["2024-01-01"]),
    "effective_to": pd.to_datetime(["2024-12-31"]),
})

result = compute_changes(
    current_state, updates,
    id_columns=["id"],
    value_columns=["value"],
    system_date=pd.Timestamp("2024-05-15"),
    update_mode="full_state",  # Enable tombstoning
)

# B is tombstoned: its effective_to is set to 2024-05-15
```

## API

### `compute_changes(current_state, updates, id_columns, value_columns, system_date=None, update_mode="delta")`

**Parameters:**
- `current_state`: DataFrame with columns `[*id_columns, *value_columns, effective_from, effective_to, as_of_from, as_of_to]`
- `updates`: DataFrame with columns `[*id_columns, *value_columns, effective_from, effective_to]`
- `id_columns`: list of column names that identify a unique timeseries
- `value_columns`: list of column names containing the data values
- `system_date`: date for `as_of_from`/`as_of_to` timestamps (defaults to today)
- `update_mode`: either `"delta"` or `"full_state"`
  - `"delta"` (default): updates overlay on current state; IDs not in updates are unchanged
  - `"full_state"`: updates represent complete state; IDs in current but not in updates are tombstoned

**Returns:** `BitemporalChanges(expires, inserts)`
- `expires`: DataFrame of records to close (includes `as_of_to` set to `system_date`)
- `inserts`: DataFrame of new records (includes `as_of_from` set to `system_date`)

## How it works

1. **Collect boundaries**: gather all `effective_from` and `effective_to` dates from both current state and updates
2. **Create segments**: split the timeline at each boundary point
3. **Resolve winners**: for each segment, pick the record with highest priority (updates > current) and latest sequence number
4. **Merge adjacent**: combine consecutive segments with identical values
5. **Diff against current**: compute expires (current records not in merged) and inserts (merged records not in current)

All of this happens in a single SQL query executed by chdb.

## Performance

### Delta Mode

Tested with 500k current state rows (125k ID groups, 4 records each) and 300k updates:

| Metric | Value |
|--------|-------|
| Computation time | ~1.2s |
| Throughput | ~650k rows/sec |
| Records expired | ~234k |
| Records inserted | ~568k |

Update mix: 40% overlapping, 20% adjacent same-value, 20% same-value no-op, 20% new IDs.

### Full State Mode

Tested with 500k current state rows (125k ID groups) and 75k updates (40% of IDs tombstoned):

| Metric | Value |
|--------|-------|
| Computation time | ~0.9s |
| Throughput | ~633k rows/sec |
| Records expired | ~269k |
| Records inserted | ~268k |
| Tombstone records | ~200k |

Full state mode adds minimal overhead despite the additional tombstone generation queries.

### Implementation Notes

The library uses chdb's temporary tables with the `Python()` table function for zero-copy DataFrame access. This approach is ~18% faster than writing to temporary Parquet files.

One quirk: `pd.NaT` values must be converted to `None` before insertion, as chdb's `Python()` function converts NaT to an invalid date rather than NULL. This conversion is handled automatically by the library.

## Running tests

```bash
uv run pytest -v
```

## Requirements

- Python 3.11+
- chdb
- pandas

## License

MIT
