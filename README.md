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

## API

### `compute_changes(current_state, updates, id_columns, value_columns, system_date=None)`

**Parameters:**
- `current_state`: DataFrame with columns `[*id_columns, *value_columns, effective_from, effective_to, as_of_from, as_of_to]`
- `updates`: DataFrame with columns `[*id_columns, *value_columns, effective_from, effective_to]`
- `id_columns`: list of column names that identify a unique timeseries
- `value_columns`: list of column names containing the data values
- `system_date`: date for `as_of_from`/`as_of_to` timestamps (defaults to today)

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

Tested with 500k current state rows (125k ID groups, 4 records each) and 300k updates:

| Metric | Value |
|--------|-------|
| Computation time | 1.53s |
| Throughput | 521,543 rows/sec |
| Records expired | 215,400 |
| Records inserted | 504,600 |

Update mix: 40% overlapping, 20% adjacent same-value, 20% same-value no-op, 20% new IDs.

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
