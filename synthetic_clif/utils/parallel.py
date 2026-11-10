"""Parallel generation utilities for synthetic CLIF datasets."""

import io
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def chunk_dataframe(df: pd.DataFrame, n_chunks: int) -> list[pd.DataFrame]:
    """Split DataFrame into N roughly equal chunks.

    Args:
        df: DataFrame to split
        n_chunks: Number of chunks to create

    Returns:
        List of DataFrame chunks
    """
    if n_chunks <= 1 or len(df) == 0:
        return [df]
    n_chunks = min(n_chunks, len(df))
    chunk_size = math.ceil(len(df) / n_chunks)
    return [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]


def generate_chunk_seeds(base_seed: int, n_chunks: int) -> list[int]:
    """Generate deterministic child seeds from a base seed.

    Args:
        base_seed: The base seed to derive children from
        n_chunks: Number of child seeds to generate

    Returns:
        List of integer seeds
    """
    rng = np.random.default_rng(base_seed)
    return [int(rng.integers(0, 2**31)) for _ in range(n_chunks)]


def _df_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    """Serialize DataFrame to parquet bytes for cross-process transfer."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def _parquet_bytes_to_df(data: bytes) -> pd.DataFrame:
    """Deserialize DataFrame from parquet bytes."""
    return pd.read_parquet(io.BytesIO(data))


def _worker_generate_chunk(
    generator_class,
    hosp_chunk_bytes: bytes,
    seed: int,
    mcide_dir: Optional[str],
    extra_df_bytes: dict[str, bytes],
) -> bytes:
    """Top-level picklable function for worker processes.

    Creates a fresh generator and MCIDELoader in the worker process,
    processes one chunk of hospitalizations, and returns parquet bytes.

    Args:
        generator_class: The generator class to instantiate (picklable by reference)
        hosp_chunk_bytes: Hospitalization DataFrame as parquet bytes
        seed: Random seed for this chunk's generator
        mcide_dir: Optional path to mCIDE CSV directory (as string)
        extra_df_bytes: Additional DataFrames needed by generate(), as parquet bytes

    Returns:
        Result DataFrame as parquet bytes, or empty bytes if no rows generated
    """
    from synthetic_clif.config.mcide import MCIDELoader

    # Reconstruct DataFrames from parquet bytes
    hosp_chunk = pd.read_parquet(io.BytesIO(hosp_chunk_bytes))
    extra_dfs = {
        name: pd.read_parquet(io.BytesIO(df_bytes))
        for name, df_bytes in extra_df_bytes.items()
    }

    # Create fresh MCIDELoader and generator in this worker process
    mcide = MCIDELoader(Path(mcide_dir) if mcide_dir else None)
    gen = generator_class(seed=seed, mcide=mcide)

    # Call generate with the chunk and any extra DataFrames as kwargs
    result = gen.generate(hosp_chunk, **extra_dfs)

    # Serialize result back to parquet bytes
    if len(result) == 0:
        return b""
    return _df_to_parquet_bytes(result)


def parallel_generate_chunked(
    generator_class,
    hosp_df: pd.DataFrame,
    n_workers: int,
    base_seed: int,
    mcide_dir: Optional[Path] = None,
    extra_dfs: Optional[dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Orchestrate parallel generation using ProcessPoolExecutor.

    Splits hospitalizations into chunks, processes each in a separate
    worker process with a fresh generator instance, then concatenates results.

    Args:
        generator_class: The generator class to use
        hosp_df: Hospitalizations DataFrame to split across workers
        n_workers: Number of worker processes
        base_seed: Base seed for generating per-chunk seeds
        mcide_dir: Optional path to mCIDE CSV directory
        extra_dfs: Additional DataFrames to pass to generate() (e.g. adt_df)

    Returns:
        Concatenated DataFrame from all workers
    """
    if extra_dfs is None:
        extra_dfs = {}

    # Split hospitalizations into chunks
    chunks = chunk_dataframe(hosp_df, n_workers)
    seeds = generate_chunk_seeds(base_seed, len(chunks))

    mcide_dir_str = str(mcide_dir) if mcide_dir else None

    futures = []
    # Vitals, labs and respiratory run this function from three threads at
    # once. Forking a multi-threaded parent can copy a lock another thread
    # holds (pyarrow's, for one) into the child, which then hangs. Start
    # workers from a clean forkserver (or spawn) process instead.
    method = "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"
    with ProcessPoolExecutor(
        max_workers=n_workers, mp_context=multiprocessing.get_context(method)
    ) as executor:
        for chunk, chunk_seed in zip(chunks, seeds):
            # Filter extra DataFrames to this chunk's hospitalization IDs
            chunk_hosp_ids = set(chunk["hospitalization_id"])
            filtered_extra_bytes = {}
            for name, df in extra_dfs.items():
                if "hospitalization_id" in df.columns:
                    filtered = df[df["hospitalization_id"].isin(chunk_hosp_ids)]
                else:
                    filtered = df
                filtered_extra_bytes[name] = _df_to_parquet_bytes(filtered)

            hosp_bytes = _df_to_parquet_bytes(chunk)

            future = executor.submit(
                _worker_generate_chunk,
                generator_class,
                hosp_bytes,
                chunk_seed,
                mcide_dir_str,
                filtered_extra_bytes,
            )
            futures.append(future)

        # Collect results
        result_dfs = []
        for future in futures:
            result_bytes = future.result()
            if result_bytes:
                result_dfs.append(_parquet_bytes_to_df(result_bytes))

    if not result_dfs:
        return pd.DataFrame()

    return pd.concat(result_dfs, ignore_index=True)
