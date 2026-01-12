"""
Bitemporal Timeseries Processing using chdb (embedded ClickHouse)

This module implements timeline reconstruction for bitemporal data:
- Handles multiple overlapping updates per ID in a single batch
- Produces expires (records to close with as_of_to) and inserts (new records)
- Uses pure SQL with array operations - no iterative slicing

Author: Claude (prototype for Mike)
"""

import chdb
import pandas as pd
import tempfile
import os
from datetime import date
from typing import NamedTuple


class BitemporalChanges(NamedTuple):
    """Result of processing bitemporal updates."""
    expires: pd.DataFrame  # Original rows to mark with as_of_to
    inserts: pd.DataFrame  # New rows to insert


def compute_changes(
    current_state: pd.DataFrame,
    updates: pd.DataFrame,
    id_columns: list[str],
    value_columns: list[str],
    system_date: date | None = None,
) -> BitemporalChanges:
    """
    Compute bitemporal changes using timeline reconstruction.
    
    Args:
        current_state: DataFrame with columns: 
            [*id_columns, *value_columns, effective_from, effective_to, as_of_from, as_of_to]
        updates: DataFrame with columns:
            [*id_columns, *value_columns, effective_from, effective_to]
        id_columns: Columns that identify a unique timeseries
        value_columns: Columns containing the actual data values
        system_date: System date for as_of timestamps (defaults to today)
    
    Returns:
        BitemporalChanges with expires and inserts DataFrames
    """
    system_date = system_date or date.today()
    
    # Write DataFrames to temporary parquet files for chdb to read
    with tempfile.TemporaryDirectory() as tmpdir:
        current_path = os.path.join(tmpdir, "current_state.parquet")
        updates_path = os.path.join(tmpdir, "updates.parquet")
        
        current_state.to_parquet(current_path)
        updates.to_parquet(updates_path)
        
        id_cols_sql = ", ".join(id_columns)
        value_cols_sql = ", ".join(value_columns)
        all_value_cols = ", ".join([f"any({c}) AS {c}" for c in value_columns])
        
        # Build hash expression for change detection
        hash_cols = ", ".join([f"toString({c})" for c in value_columns])
        hash_expr = f"xxHash64(concat({hash_cols}))"
        
        # File references for chdb
        current_file = f"file('{current_path}', Parquet)"
        updates_file = f"file('{updates_path}', Parquet)"
        
        # The core CTEs implement timeline reconstruction
        core_ctes = f"""
        WITH 
        -- Tag current state with source and priority
        current_tagged AS (
            SELECT 
                {id_cols_sql},
                {value_cols_sql},
                effective_from,
                effective_to,
                as_of_from,
                as_of_to,
                {hash_expr} AS value_hash,
                'current' AS source,
                0 AS priority,
                0 AS update_seq,
                CAST(rowNumberInAllBlocks() AS Int64) AS original_row_id
            FROM {current_file}
            WHERE as_of_to IS NULL  -- Only consider active records
        ),
        
        -- Tag updates with source and priority
        updates_tagged AS (
            SELECT 
                {id_cols_sql},
                {value_cols_sql},
                effective_from,
                effective_to,
                CAST(NULL AS Nullable(Date)) AS as_of_from,
                CAST(NULL AS Nullable(Date)) AS as_of_to,
                {hash_expr} AS value_hash,
                'update' AS source,
                1 AS priority,
                rowNumberInAllBlocks() AS update_seq,
                CAST(-1 AS Int64) AS original_row_id
            FROM {updates_file}
        ),
        
        -- Combine all records
        all_records AS (
            SELECT * FROM current_tagged
            UNION ALL
            SELECT * FROM updates_tagged
        ),
        
        -- Collect all unique time boundaries per ID group
        boundaries AS (
            SELECT 
                {id_cols_sql},
                arraySort(arrayDistinct(
                    arrayFlatten(
                        groupArray([effective_from, effective_to])
                    )
                )) AS cuts
            FROM all_records
            GROUP BY {id_cols_sql}
        ),
        
        -- Explode into segments (consecutive boundary pairs)
        segments AS (
            SELECT 
                {id_cols_sql},
                segment.1 AS seg_from,
                segment.2 AS seg_to
            FROM boundaries
            ARRAY JOIN arrayZip(
                arraySlice(cuts, 1, -1),
                arraySlice(cuts, 2)
            ) AS segment
            WHERE segment.1 < segment.2  -- Filter zero-width segments
        ),
        
        -- For each segment, find all records that cover it
        segment_candidates AS (
            SELECT 
                s.{id_cols_sql.replace(', ', ', s.')},
                s.seg_from,
                s.seg_to,
                r.value_hash,
                r.source,
                r.priority,
                r.update_seq,
                r.original_row_id,
                {', '.join([f'r.{c}' for c in value_columns])}
            FROM segments s
            INNER JOIN all_records r 
                ON ({' AND '.join([f's.{c} = r.{c}' for c in id_columns])})
                AND r.effective_from <= s.seg_from 
                AND r.effective_to >= s.seg_to
        ),
        
        -- Pick the winner for each segment (highest priority, then latest update_seq)
        segment_winners AS (
            SELECT 
                {id_cols_sql},
                seg_from,
                seg_to,
                value_hash,
                source,
                original_row_id,
                {value_cols_sql}
            FROM (
                SELECT 
                    *,
                    row_number() OVER (
                        PARTITION BY {id_cols_sql}, seg_from, seg_to 
                        ORDER BY priority DESC, update_seq DESC
                    ) AS rn
                FROM segment_candidates
            )
            WHERE rn = 1
        ),
        
        -- Merge adjacent segments with same value_hash into contiguous ranges
        merged AS (
            SELECT
                {id_cols_sql},
                min(seg_from) AS effective_from,
                max(seg_to) AS effective_to,
                value_hash,
                any(source) AS source,
                {all_value_cols}
            FROM (
                SELECT
                    *,
                    sum(is_boundary) OVER (
                        PARTITION BY {id_cols_sql} 
                        ORDER BY seg_from 
                        ROWS UNBOUNDED PRECEDING
                    ) AS group_id
                FROM (
                    SELECT
                        *,
                        CASE 
                            WHEN value_hash != lagInFrame(value_hash, 1, value_hash) OVER (
                                PARTITION BY {id_cols_sql} ORDER BY seg_from
                            )
                            OR seg_from != lagInFrame(seg_to, 1, seg_from) OVER (
                                PARTITION BY {id_cols_sql} ORDER BY seg_from
                            )
                            THEN 1 
                            ELSE 0 
                        END AS is_boundary
                    FROM segment_winners
                )
            )
            GROUP BY {id_cols_sql}, value_hash, group_id
        ),
        
        -- Find current records that have an exact match in merged (no change needed)
        exact_matches AS (
            SELECT c.original_row_id
            FROM current_tagged c
            INNER JOIN merged m 
                ON ({' AND '.join([f'c.{col} = m.{col}' for col in id_columns])})
                AND c.effective_from = m.effective_from
                AND c.effective_to = m.effective_to
                AND c.value_hash = m.value_hash
        ),
        
        -- Records to expire = current records without an exact match
        to_expire AS (
            SELECT original_row_id
            FROM current_tagged
            WHERE original_row_id NOT IN (SELECT original_row_id FROM exact_matches)
        ),
        
        -- Find merged records that match current exactly (to exclude from inserts)
        merged_matches AS (
            SELECT 
                {', '.join([f'm.{col}' for col in id_columns])},
                m.effective_from,
                m.effective_to,
                m.value_hash
            FROM merged m
            INNER JOIN current_tagged c
                ON ({' AND '.join([f'm.{col} = c.{col}' for col in id_columns])})
                AND m.effective_from = c.effective_from
                AND m.effective_to = c.effective_to
                AND m.value_hash = c.value_hash
        ),
        
        -- Find new records to insert (from merged that don't match current exactly)
        to_insert AS (
            SELECT 
                {id_cols_sql},
                {value_cols_sql},
                effective_from,
                effective_to,
                value_hash
            FROM merged
            WHERE ({id_cols_sql}, effective_from, effective_to, value_hash) NOT IN (
                SELECT {id_cols_sql}, effective_from, effective_to, value_hash FROM merged_matches
            )
        )
        """
        
        # Query for expires
        expires_value_cols = ", ".join([f"c.{v}" for v in value_columns])
        expires_query = core_ctes + f"""
        SELECT 
            c.{id_cols_sql.replace(', ', ', c.')},
            {expires_value_cols},
            c.effective_from,
            c.effective_to,
            c.as_of_from,
            toDate('{system_date}') AS as_of_to
        FROM current_tagged c
        WHERE c.original_row_id IN (SELECT original_row_id FROM to_expire)
        """
        
        # Query for inserts
        inserts_query = core_ctes + f"""
        SELECT 
            {id_cols_sql},
            {value_cols_sql},
            effective_from,
            effective_to,
            toDate('{system_date}') AS as_of_from,
            CAST(NULL AS Nullable(Date)) AS as_of_to
        FROM to_insert
        ORDER BY {id_cols_sql}, effective_from
        """
        
        # Execute queries
        expires_result = chdb.query(expires_query, output_format="DataFrame")
        inserts_result = chdb.query(inserts_query, output_format="DataFrame")
        
        return BitemporalChanges(
            expires=expires_result if expires_result is not None else pd.DataFrame(),
            inserts=inserts_result if inserts_result is not None else pd.DataFrame(),
        )
