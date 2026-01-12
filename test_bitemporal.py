"""
Tests for Bitemporal Timeseries Processing
"""

import pytest
import pandas as pd
import numpy as np
import time
from bitemporal import compute_changes


class TestSimpleUpdates:
    """Tests for basic update scenarios."""

    def test_single_update_slices_existing_record(self):
        """Single update should slice through existing record into 3 segments."""
        current = pd.DataFrame({
            "id": ["A"],
            "value": [100],
            "effective_from": pd.to_datetime(["2024-01-01"]),
            "effective_to": pd.to_datetime(["2024-12-31"]),
            "as_of_from": pd.to_datetime(["2024-01-01"]),
            "as_of_to": pd.Series([pd.NaT]),
        })

        updates = pd.DataFrame({
            "id": ["A"],
            "value": [200],
            "effective_from": pd.to_datetime(["2024-06-01"]),
            "effective_to": pd.to_datetime(["2024-08-31"]),
        })

        result = compute_changes(
            current, updates,
            id_columns=["id"],
            value_columns=["value"],
            system_date=pd.Timestamp("2024-05-15"),
        )

        # Original record should expire
        assert len(result.expires) == 1
        assert pd.Timestamp(result.expires.iloc[0]["as_of_to"]) == pd.Timestamp("2024-05-15")

        # Should produce 3 new records
        assert len(result.inserts) == 3

        inserts = result.inserts.sort_values("effective_from").reset_index(drop=True)

        # First segment: original value before update
        assert pd.Timestamp(inserts.iloc[0]["effective_from"]).date() == pd.Timestamp("2024-01-01").date()
        assert pd.Timestamp(inserts.iloc[0]["effective_to"]).date() == pd.Timestamp("2024-06-01").date()
        assert inserts.iloc[0]["value"] == 100

        # Second segment: update value
        assert pd.Timestamp(inserts.iloc[1]["effective_from"]).date() == pd.Timestamp("2024-06-01").date()
        assert pd.Timestamp(inserts.iloc[1]["effective_to"]).date() == pd.Timestamp("2024-08-31").date()
        assert inserts.iloc[1]["value"] == 200

        # Third segment: original value after update
        assert pd.Timestamp(inserts.iloc[2]["effective_from"]).date() == pd.Timestamp("2024-08-31").date()
        assert pd.Timestamp(inserts.iloc[2]["effective_to"]).date() == pd.Timestamp("2024-12-31").date()
        assert inserts.iloc[2]["value"] == 100

    def test_update_with_same_value_no_change(self):
        """Update with same value should result in no effective change."""
        current = pd.DataFrame({
            "id": ["A"],
            "value": [100],
            "effective_from": pd.to_datetime(["2024-01-01"]),
            "effective_to": pd.to_datetime(["2024-12-31"]),
            "as_of_from": pd.to_datetime(["2024-01-01"]),
            "as_of_to": pd.Series([pd.NaT]),
        })

        updates = pd.DataFrame({
            "id": ["A"],
            "value": [100],  # Same value
            "effective_from": pd.to_datetime(["2024-03-01"]),
            "effective_to": pd.to_datetime(["2024-06-01"]),
        })

        result = compute_changes(
            current, updates,
            id_columns=["id"],
            value_columns=["value"],
            system_date=pd.Timestamp("2024-05-15"),
        )

        # No expires needed (or reconstruct same record)
        # The merged timeline should match the original
        if len(result.expires) == 0:
            assert len(result.inserts) == 0
        else:
            # If it expires and reinserts, should be equivalent
            assert len(result.inserts) == 1
            assert pd.Timestamp(result.inserts.iloc[0]["effective_from"]) == pd.Timestamp("2024-01-01")
            assert pd.Timestamp(result.inserts.iloc[0]["effective_to"]) == pd.Timestamp("2024-12-31")
            assert result.inserts.iloc[0]["value"] == 100


class TestOverlappingUpdates:
    """Tests for overlapping update scenarios."""

    def test_multiple_overlapping_updates_same_id(self):
        """Later update should win in overlapping region."""
        current = pd.DataFrame({
            "id": ["A"],
            "value": [100],
            "effective_from": pd.to_datetime(["2024-01-01"]),
            "effective_to": pd.to_datetime(["2024-12-31"]),
            "as_of_from": pd.to_datetime(["2024-01-01"]),
            "as_of_to": pd.Series([pd.NaT]),
        })

        # Two updates that overlap from May-July
        updates = pd.DataFrame({
            "id": ["A", "A"],
            "value": [200, 300],
            "effective_from": pd.to_datetime(["2024-03-01", "2024-05-01"]),
            "effective_to": pd.to_datetime(["2024-07-31", "2024-09-30"]),
        })

        result = compute_changes(
            current, updates,
            id_columns=["id"],
            value_columns=["value"],
            system_date=pd.Timestamp("2024-05-15"),
        )

        assert len(result.expires) == 1

        # Should have 4 segments
        assert len(result.inserts) == 4

        inserts = result.inserts.sort_values("effective_from").reset_index(drop=True)

        # [2024-01-01 to 2024-03-01, value=100]
        assert inserts.iloc[0]["value"] == 100
        # [2024-03-01 to 2024-05-01, value=200]
        assert inserts.iloc[1]["value"] == 200
        # [2024-05-01 to 2024-09-30, value=300] - later update wins
        assert inserts.iloc[2]["value"] == 300
        # [2024-09-30 to 2024-12-31, value=100]
        assert inserts.iloc[3]["value"] == 100

    def test_adjacent_same_value_should_merge(self):
        """Adjacent segments with same value should merge into one."""
        current = pd.DataFrame({
            "id": ["A"],
            "value": [100],
            "effective_from": pd.to_datetime(["2024-01-01"]),
            "effective_to": pd.to_datetime(["2024-12-31"]),
            "as_of_from": pd.to_datetime(["2024-01-01"]),
            "as_of_to": pd.Series([pd.NaT]),
        })

        # Two adjacent updates with same value
        updates = pd.DataFrame({
            "id": ["A", "A"],
            "value": [200, 200],
            "effective_from": pd.to_datetime(["2024-03-01", "2024-06-01"]),
            "effective_to": pd.to_datetime(["2024-06-01", "2024-09-01"]),
        })

        result = compute_changes(
            current, updates,
            id_columns=["id"],
            value_columns=["value"],
            system_date=pd.Timestamp("2024-05-15"),
        )

        # Should have 3 segments (merged middle)
        assert len(result.inserts) == 3

        inserts = result.inserts.sort_values("effective_from").reset_index(drop=True)

        # Middle segment should be merged
        assert pd.Timestamp(inserts.iloc[1]["effective_from"]).date() == pd.Timestamp("2024-03-01").date()
        assert pd.Timestamp(inserts.iloc[1]["effective_to"]).date() == pd.Timestamp("2024-09-01").date()
        assert inserts.iloc[1]["value"] == 200


class TestMultipleIds:
    """Tests for multi-ID scenarios."""

    def test_multiple_ids_in_batch(self):
        """Multiple IDs in same batch should be handled independently."""
        current = pd.DataFrame({
            "id": ["A", "B"],
            "value": [100, 500],
            "effective_from": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "effective_to": pd.to_datetime(["2024-12-31", "2024-12-31"]),
            "as_of_from": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "as_of_to": pd.Series([pd.NaT, pd.NaT]),
        })

        updates = pd.DataFrame({
            "id": ["A", "B"],
            "value": [200, 600],
            "effective_from": pd.to_datetime(["2024-06-01", "2024-03-01"]),
            "effective_to": pd.to_datetime(["2024-09-01", "2024-06-01"]),
        })

        result = compute_changes(
            current, updates,
            id_columns=["id"],
            value_columns=["value"],
            system_date=pd.Timestamp("2024-05-15"),
        )

        # Both original records should expire
        assert len(result.expires) == 2

        # Each ID should have 3 segments
        inserts_a = result.inserts[result.inserts["id"] == "A"]
        inserts_b = result.inserts[result.inserts["id"] == "B"]

        assert len(inserts_a) == 3
        assert len(inserts_b) == 3

    def test_composite_id_columns(self):
        """Composite ID columns should work correctly."""
        current = pd.DataFrame({
            "portfolio": ["P1", "P1"],
            "security": ["AAPL", "MSFT"],
            "quantity": [100, 200],
            "effective_from": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "effective_to": pd.to_datetime(["2024-12-31", "2024-12-31"]),
            "as_of_from": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "as_of_to": pd.Series([pd.NaT, pd.NaT]),
        })

        updates = pd.DataFrame({
            "portfolio": ["P1"],
            "security": ["AAPL"],
            "quantity": [150],
            "effective_from": pd.to_datetime(["2024-06-01"]),
            "effective_to": pd.to_datetime(["2024-09-01"]),
        })

        result = compute_changes(
            current, updates,
            id_columns=["portfolio", "security"],
            value_columns=["quantity"],
            system_date=pd.Timestamp("2024-05-15"),
        )

        # Only AAPL should expire
        assert len(result.expires) == 1
        assert result.expires.iloc[0]["security"] == "AAPL"

        # AAPL should have 3 new segments, MSFT unchanged
        inserts_aapl = result.inserts[result.inserts["security"] == "AAPL"]
        inserts_msft = result.inserts[result.inserts["security"] == "MSFT"]

        assert len(inserts_aapl) == 3
        assert len(inserts_msft) == 0  # MSFT not touched


class TestPerformance:
    """Performance tests with large datasets."""

    def test_large_current_state_with_mixed_updates(self):
        """
        Performance test with 500k current rows and 300k updates.

        Current state:
        - ~125k unique ID groups (portfolio + security)
        - Each ID group has 4 consecutive records spanning ~4 years
        - Records are contiguous (no gaps in effective time)

        Updates are a mixture of:
        - 40% overlapping updates (slice through existing records)
        - 20% adjacent same-value (should merge)
        - 20% same-value no-op (should produce minimal changes)
        - 20% new ID groups (extend coverage)
        """
        np.random.seed(42)

        # Configuration
        num_portfolios = 500
        num_securities = 250
        records_per_id = 4
        num_id_groups = 125_000  # Will give us 500k current rows
        num_updates = 300_000

        base_date = pd.Timestamp("2020-01-01")
        days_per_record = 365  # Each record covers ~1 year

        print(f"\nGenerating test data...")
        gen_start = time.perf_counter()

        # Generate ID groups as cartesian product subset
        # 500 portfolios x 250 securities = 125k unique combinations
        id_group_portfolios = []
        id_group_securities = []
        for p in range(num_portfolios):
            for s in range(num_securities):
                id_group_portfolios.append(f"P{p:04d}")
                id_group_securities.append(f"SEC{s:05d}")

        # Build current state: 4 contiguous records per ID group
        current_portfolios = []
        current_securities = []
        current_quantities = []
        current_prices = []
        current_from = []
        current_to = []

        for i in range(num_id_groups):
            portfolio = id_group_portfolios[i]
            security = id_group_securities[i]

            # Generate 4 consecutive records with different values
            for j in range(records_per_id):
                current_portfolios.append(portfolio)
                current_securities.append(security)
                current_quantities.append(100 * (j + 1) + i % 100)
                current_prices.append(round(10.0 + j * 5.0 + (i % 50) * 0.1, 2))
                current_from.append(base_date + pd.Timedelta(days=j * days_per_record))
                current_to.append(base_date + pd.Timedelta(days=(j + 1) * days_per_record))

        current_state = pd.DataFrame({
            "portfolio": current_portfolios,
            "security": current_securities,
            "quantity": current_quantities,
            "price": current_prices,
            "effective_from": current_from,
            "effective_to": current_to,
            "as_of_from": [base_date] * len(current_portfolios),
            "as_of_to": pd.Series([pd.NaT] * len(current_portfolios)),
        })

        # Generate updates with different scenarios
        update_portfolios = []
        update_securities = []
        update_quantities = []
        update_prices = []
        update_from = []
        update_to = []

        # Scenario counts
        n_overlapping = int(num_updates * 0.4)      # 40% overlapping
        n_adjacent_same = int(num_updates * 0.2)    # 20% adjacent same value
        n_same_value = int(num_updates * 0.2)       # 20% same value no-op
        n_new_ids = num_updates - n_overlapping - n_adjacent_same - n_same_value  # 20% new

        # 1. Overlapping updates - slice through existing records
        for i in range(n_overlapping):
            idx = i % num_id_groups
            update_portfolios.append(id_group_portfolios[idx])
            update_securities.append(id_group_securities[idx])
            # Different value to force a change
            update_quantities.append(9999 - (i % 1000))
            update_prices.append(round(999.0 - (i % 100) * 0.5, 2))
            # Overlap across record boundaries (e.g., days 300-500 crosses first two records)
            start_day = 300 + (i % 400)
            update_from.append(base_date + pd.Timedelta(days=start_day))
            update_to.append(base_date + pd.Timedelta(days=start_day + 200))

        # 2. Adjacent same value - should trigger merging
        for i in range(n_adjacent_same):
            idx = i % num_id_groups
            update_portfolios.append(id_group_portfolios[idx])
            update_securities.append(id_group_securities[idx])
            # Same value as what we'll insert adjacently
            update_quantities.append(5000)
            update_prices.append(50.0)
            # Two adjacent ranges that should merge
            if i % 2 == 0:
                update_from.append(base_date + pd.Timedelta(days=100))
                update_to.append(base_date + pd.Timedelta(days=200))
            else:
                update_from.append(base_date + pd.Timedelta(days=200))
                update_to.append(base_date + pd.Timedelta(days=300))

        # 3. Same value updates - should produce minimal/no changes
        for i in range(n_same_value):
            idx = i % num_id_groups
            update_portfolios.append(id_group_portfolios[idx])
            update_securities.append(id_group_securities[idx])
            # Match existing value for first record of this ID
            record_idx = idx * records_per_id
            update_quantities.append(current_quantities[record_idx])
            update_prices.append(current_prices[record_idx])
            # Subset of first record's range
            update_from.append(base_date + pd.Timedelta(days=50))
            update_to.append(base_date + pd.Timedelta(days=200))

        # 4. New ID groups - IDs not in current state
        for i in range(n_new_ids):
            # Use high portfolio numbers not in current state
            update_portfolios.append(f"PNEW{i:05d}")
            update_securities.append(f"SECNEW{i:05d}")
            update_quantities.append(1000 + i % 500)
            update_prices.append(round(100.0 + (i % 100), 2))
            update_from.append(base_date + pd.Timedelta(days=i % 1000))
            update_to.append(base_date + pd.Timedelta(days=(i % 1000) + 365))

        updates = pd.DataFrame({
            "portfolio": update_portfolios,
            "security": update_securities,
            "quantity": update_quantities,
            "price": update_prices,
            "effective_from": update_from,
            "effective_to": update_to,
        })

        gen_elapsed = time.perf_counter() - gen_start

        # Stats
        current_combos = set(zip(current_state["portfolio"], current_state["security"]))
        update_combos = set(zip(updates["portfolio"], updates["security"]))

        print(f"Data generation: {gen_elapsed:.2f}s")
        print(f"Current state: {len(current_state):,} rows, {len(current_combos):,} ID groups")
        print(f"Updates: {len(updates):,} rows, {len(update_combos):,} unique IDs")
        print(f"  - Overlapping: {n_overlapping:,}")
        print(f"  - Adjacent same-value: {n_adjacent_same:,}")
        print(f"  - Same-value no-op: {n_same_value:,}")
        print(f"  - New IDs: {n_new_ids:,}")

        # Run computation
        print(f"\nRunning compute_changes...")
        start_time = time.perf_counter()

        result = compute_changes(
            current_state, updates,
            id_columns=["portfolio", "security"],
            value_columns=["quantity", "price"],
            system_date=pd.Timestamp("2024-01-15"),
        )

        elapsed = time.perf_counter() - start_time

        # Report results
        print(f"\n{'='*60}")
        print(f"Performance Test Results")
        print(f"{'='*60}")
        print(f"Computation time: {elapsed:.2f}s")
        print(f"Current records: {len(current_state):,}")
        print(f"Update records: {len(updates):,}")
        print(f"Records expired: {len(result.expires):,}")
        print(f"Records inserted: {len(result.inserts):,}")
        print(f"Throughput: {(len(current_state) + len(updates)) / elapsed:,.0f} rows/sec")
        print(f"{'='*60}")

        # Assertions
        assert len(result.expires) >= 0
        assert len(result.inserts) >= 0
        assert elapsed < 120, f"Performance regression: took {elapsed:.2f}s"
