from pathlib import Path

import pandas as pd


def main() -> None:
    """Load the router test parquet file and print the first five rows."""
    data_path = Path("dataset") / "rlla_4k" / "test_router.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find parquet file at {data_path}")

    df = pd.read_parquet(data_path)
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(df.head())


if __name__ == "__main__":
    main()


