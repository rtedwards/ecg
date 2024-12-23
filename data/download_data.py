import kagglehub
from pathlib import Path
from ecg import PACKAGE_ROOT
import polars as pl

# Download latest version
download_path = Path(kagglehub.dataset_download("shayanfazeli/heartbeat"))

directory = Path(PACKAGE_ROOT) / "data"

print("Downloading files to:", directory)

files = ["mitbih_train.csv", "mitbih_test.csv", "ptbdb_abnormal.csv", "ptbdb_normal.csv"]
target = "column_188"
for file in files:
    df = pl.read_csv(download_path / file, has_header=False).rename({"column_188": "target"})
    df.write_parquet((directory / file).with_suffix(".parquet"))

for file in list(directory.glob("*.parquet")):
    print(file)
