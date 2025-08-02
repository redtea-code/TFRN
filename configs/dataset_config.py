import pandas as pd
from pathlib import Path


class DatasetConfig:
    dataset = "AUS"
    camels_root = Path(r"D:\RR-Former\camels_aus") # your CAMELS-AUS dataset root
    basins_file = "data/aus_561basins_list.txt"
    forcing_type = "AUS"
    basin_mark = "561"


    # dataset = "US"
    # camels_root = Path(r"D:\ealstm\CAMELS_US")  # your CAMELS dataset root
    # forcing_type = "daymet"  # TODO: "daymet" or "maurer_extended" or "nldas_extended" or AUS
    # basin_mark = "671"  # TODO: daymet in [673, 671], maurer_extended in [448]
    # basins_file = f"data/671basins_list.txt"

    decompose = None
    global_basins_list = pd.read_csv(basins_file, header=None, dtype=str)[0].values.tolist()

    # TODO: daymet date
    train_start = pd.to_datetime("1980-10-01", format="%Y-%m-%d")
    train_end = pd.to_datetime("1995-09-30", format="%Y-%m-%d")
    val_start = pd.to_datetime("1995-10-01", format="%Y-%m-%d")
    val_end = pd.to_datetime("2000-09-30", format="%Y-%m-%d")
    test_start = pd.to_datetime("2000-10-01", format="%Y-%m-%d")
    test_end = pd.to_datetime("2014-09-30", format="%Y-%m-%d")

    dataset_info = f"{forcing_type}{basin_mark}_{train_start.year}~{train_end.year}#{val_start.year}~{val_end.year}#{test_start.year}~{test_end.year}"
