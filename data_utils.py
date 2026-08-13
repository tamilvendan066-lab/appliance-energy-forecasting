"""
data_utils.py
=============
Part 1: I use this module to retrieve and prepare the Appliances Energy
Prediction dataset - downloading it, parsing timestamps, resampling to
hourly, and checking for missing values.

Functions
---------
download_data          -> I fetch the raw CSV from the UCI repository (with retries)
load_raw_data           -> I read the CSV, parse timestamps, and set the index
resample_hourly         -> I bin the 10-minute data up to hourly means
check_missing_values    -> I report and handle any missing values after resampling
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np
import requests

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00374/energydata_complete.csv"
)


def download_data(
    url: str = DATA_URL,
    dest_path: str = "data/energydata_complete.csv",
    max_retries: int = 5,
) -> str:
    """
    I download the raw CSV file if it isn't already sitting on disk.

    I use `requests` with a streamed download and automatic retries, since
    I found the UCI server occasionally drops the connection mid-transfer
    (chunked-encoding IncompleteRead errors) - `urllib.request.urlretrieve`
    has no retry logic and fails hard on the first hiccup, so I avoid it.

    Parameters
    ----------
    url : str
        Source URL for the dataset (UCI repository).
    dest_path : str
        Local path I save the CSV to.
    max_retries : int
        Number of download attempts I allow before giving up.

    Returns
    -------
    str
        The local path of the downloaded (or already-existing) file.
    """
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

    # A genuinely complete file is ~19MB. If I find anything much smaller
    # already on disk, I treat it as a truncated leftover from a previous
    # failed attempt rather than trusting it blindly.
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1_000_000:
        print(f"Found existing complete file at {dest_path}, skipping download.")
        return dest_path

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Downloading dataset from {url} (attempt {attempt}/{max_retries}) ...")
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            size = os.path.getsize(dest_path)
            if size < 1_000_000:
                raise IOError(f"Downloaded file too small ({size} bytes) - likely truncated.")

            print(f"Saved to {dest_path} ({size:,} bytes)")
            return dest_path

        except Exception as e:
            last_error = e
            print(f"  Attempt {attempt} failed: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)

    raise RuntimeError(
        f"Download failed after {max_retries} attempts. Last error: {last_error}\n"
        f"If this keeps happening, I'd download the CSV manually in a browser from:\n"
        f"  {url}\n"
        f"and save it to: {dest_path}"
    )


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """
    I load the raw 10-minute-sampled CSV and parse the timestamp column.

    Parameters
    ----------
    csv_path : str
        Path to the downloaded CSV.

    Returns
    -------
    pd.DataFrame
        Indexed by a DatetimeIndex named 'date', all other columns numeric.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    # I leave rv1/rv2 in the raw frame here - they're random variables the
    # dataset authors included for feature-selection testing, not real
    # sensor readings, so I deliberately exclude them later when I build
    # my feature set (see feature_engineering.py).
    return df


def resample_hourly(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    """
    I bin the 10-minute data up to hourly values, as Part 1 asks for.

    Parameters
    ----------
    df : pd.DataFrame
        Raw 10-minute data, DatetimeIndex.
    how : str
        Aggregation method - I use 'mean', which is standard for sensor/
        energy data (summing wouldn't make sense for temperature/humidity).

    Returns
    -------
    pd.DataFrame
        Hourly-resampled dataframe.
    """
    if how == "mean":
        hourly = df.resample("1h").mean()
    elif how == "sum":
        hourly = df.resample("1h").sum()
    else:
        raise ValueError("how must be 'mean' or 'sum'")
    return hourly


def check_missing_values(df: pd.DataFrame, fill_method: str = "interpolate") -> pd.DataFrame:
    """
    I report missing values after resampling and fill them.

    I resample evenly-spaced 10-minute data, so I don't expect resampling
    itself to introduce gaps unless the original series already had missing
    timestamps - I make that check explicit and auditable here, rather than
    silently filling without reporting it.

    Parameters
    ----------
    df : pd.DataFrame
    fill_method : str
        'interpolate' (time-based linear interpolation) or 'ffill'.

    Returns
    -------
    pd.DataFrame
        Dataframe with missing values handled.
    """
    n_missing = df.isna().sum().sum()
    print(f"Missing values found after resampling: {n_missing}")
    if n_missing == 0:
        return df

    if fill_method == "interpolate":
        df = df.interpolate(method="time")
    elif fill_method == "ffill":
        df = df.ffill()
    else:
        raise ValueError("fill_method must be 'interpolate' or 'ffill'")

    # I backfill anything still missing (e.g. if the very first row was
    # missing, interpolation alone can't fill it).
    df = df.bfill()
    return df


def prepare_dataset(url: str = DATA_URL, dest_path: str = "data/energydata_complete.csv") -> pd.DataFrame:
    """
    My convenience wrapper: download -> load -> resample -> clean.

    Returns
    -------
    pd.DataFrame
        Clean, hourly-resampled dataset ready for my EDA.
    """
    path = download_data(url, dest_path)
    raw = load_raw_data(path)
    hourly = resample_hourly(raw)
    clean = check_missing_values(hourly)
    return clean


if __name__ == "__main__":
    data = prepare_dataset()
    print(data.head())
    print(data.shape)
