"""
history.py
==========
Lightweight persistence layer for MedBuddy AI's analysis history.

Uses a plain CSV file on disk so history survives app restarts without
requiring a database. Safe to call repeatedly; creates the file on first
write and tolerates a missing/corrupt file on read.
"""

import os
import pandas as pd

HISTORY_FILE = "history.csv"

HISTORY_COLUMNS = [
    "timestamp",
    "age",
    "gender",
    "symptoms",
    "prediction",
    "confidence",
    "confidence_level",
    "safe_to_predict",
    "threshold_used",
]


def load_history() -> pd.DataFrame:
    """
    Load analysis history from disk as a DataFrame.

    Returns an empty (but correctly-columned) DataFrame if no history
    file exists yet, or if the file is missing/corrupted.
    """

    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    try:
        history = pd.read_csv(HISTORY_FILE)

        # Guard against a manually-edited or partially-written file
        for column in HISTORY_COLUMNS:
            if column not in history.columns:
                history[column] = None

        return history[HISTORY_COLUMNS]

    except Exception:
        return pd.DataFrame(columns=HISTORY_COLUMNS)


def save_entry(entry: dict) -> None:
    """
    Append a single analysis entry to the history file.

    `entry` should contain (at least) the keys in HISTORY_COLUMNS;
    missing keys are filled with None.
    """

    history = load_history()

    row = {column: entry.get(column) for column in HISTORY_COLUMNS}
    new_row = pd.DataFrame([row], columns=HISTORY_COLUMNS)

    history = pd.concat([history, new_row], ignore_index=True)
    history.to_csv(HISTORY_FILE, index=False)


def clear_history() -> None:
    """Delete all stored history."""

    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
