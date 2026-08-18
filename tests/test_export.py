"""CSV export helpers.

Covers stable allocation rendering, so that repeated runs produce
byte-identical rows, and row writing where later rows introduce fields the
earlier ones lacked.
"""

from __future__ import annotations

import csv

from auctionlab.experiments.export import (
    allocation_to_str,
    write_csv,
    write_csv_variable_rows,
)


def test_allocation_to_str_is_stable():
    allocation = {
        "i2": frozenset({"B"}),
        "i1": frozenset({"C", "A"}),
    }
    assert allocation_to_str(allocation) == "i1:[A,C];i2:[B]"


def test_write_csv_writes_rows(tmp_path):
    path = tmp_path / "rows.csv"
    write_csv([{"a": 1, "b": 2}, {"a": 3, "b": 4}], path)
    with path.open(newline="") as handle:
        assert list(csv.DictReader(handle)) == [
            {"a": "1", "b": "2"},
            {"a": "3", "b": "4"},
        ]


def test_write_csv_variable_rows_unions_fields(tmp_path):
    path = tmp_path / "rows.csv"
    write_csv_variable_rows([{"a": 1}, {"b": 2}], path)
    with path.open(newline="") as handle:
        assert list(csv.DictReader(handle)) == [
            {"a": "1", "b": ""},
            {"a": "", "b": "2"},
        ]


def test_write_csv_empty_creates_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    write_csv([], path)
    assert path.read_text() == ""
