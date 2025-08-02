import numpy as np
import time
from bisect import bisect_right

import pandas as pd
from camels_aus.repository import CamelsAus
import torch
from pyts.decomposition import SingularSpectrumAnalysis
from sklearn.decomposition import PCA
from torch.utils.data import Dataset

# from models.utils.MEMD_ds.pyemd import select_imfs_by_variance
# from models.utils.MVMD.mvmd_python import mvmd
from statsmodels.tsa.seasonal import STL

# from data.dataset import HydroReaderFactory

"""----"""

# class Camelsaus_Dataset(Dataset):
#     def __init__(self, camels_root: str, forcing_type: str, basins_list: list, past_len: int, pred_len: int, stage: str,
#                  dates: list, x_mean=None, y_mean=None, x_std=None, y_std=None, y_stds_dict=None):
#         """Initialization
#         x_mean, y_mean, x_std, y_std should be provided if stage != "train".
#         """
#         self.camels_root = camels_root
#         self.basins_list = basins_list
#         self.past_len = past_len
#         self.pred_len = pred_len
#         self.stage = stage
#         self.dates = dates
#         self.x_dict = dict()
#         self.y_dict = dict()
#         self.date_index_dict = dict()
#         self.length_ls = list()
#
#         if y_stds_dict is None:
#             self.y_stds_dict = dict()
#         else:
#             self.y_stds_dict = y_stds_dict
#
#         self._load_data(forcing_type)
#         # Calculate mean and std
#         if self.stage == 'train':
#             self.x_mean, self.x_std = self.calc_mean_and_std(self.x_dict)
#             self.y_mean, self.y_std = self.calc_mean_and_std(self.y_dict)
#         else:
#             self.x_mean = x_mean
#             self.y_mean = y_mean
#             self.x_std = x_std
#             self.y_std = y_std
#         self.normalize_data()
#
#         self.num_samples = 0
#         for item in self.length_ls:
#             self.num_samples += item
#
#         self.index_ls = [0]
#         for i in range(len(self.length_ls)):
#             v = self.index_ls[i] + self.length_ls[i]
#             self.index_ls.append(v)
#
#     def __len__(self):
#         return self.num_samples
#
#     def __getitem__(self, idx: int):
#         basin_idx = bisect_right(self.index_ls, idx) - 1
#         local_idx = idx - self.index_ls[basin_idx]
#         basin = self.basins_list[basin_idx]
#         x_seq = self.x_dict[basin][local_idx: local_idx + self.past_len + self.pred_len, :]
#         y_seq_past = self.y_dict[basin][local_idx: local_idx + self.past_len, :]
#         y_seq_future = self.y_dict[basin][local_idx + self.past_len: local_idx + self.past_len + self.pred_len, :]
#
#         return x_seq, y_seq_past, y_seq_future, self.y_stds_dict[basin]
#
#     def _load_data(self, forcing_type):
#         # Loading vanilla data
#         basin_number = len(self.basins_list)
#         for idx, basin in enumerate(self.basins_list):
#             print(self.stage, f"{basin}: loading data %.4f" % (idx / basin_number))
#
#             reader = HydroReaderFactory.get_hydro_reader(self.camels_root, forcing_type, basin)
#             df_x = reader.get_df_x()
#             df_y = reader.get_df_y()
#
#             # Select date
#             df_x = df_x[self.dates[0]:self.dates[1]]
#             df_y = df_y[self.dates[0]:self.dates[1]]
#             assert len(df_x) == len(df_y)
#             self.date_index_dict[basin] = df_x.index
#
#             # Select used features and discharge
#             x = df_x.values.astype("float32")
#             y = df_y.values.astype("float32")
#             self.x_dict[basin] = x
#             self.y_dict[basin] = y
#
#             self.length_ls.append(len(x) - self.past_len - self.pred_len + 1)
#             # Calculate mean and std in training stage
#             if self.stage == 'train':
#                 self.y_stds_dict[basin] = y.std(axis=0).item()
#
#     @staticmethod
#     def calc_mean_and_std(data_dict):
#         data_all = np.concatenate(list(data_dict.values()), axis=0)  # CAN NOT serializable
#         nan_mean = np.nanmean(data_all, axis=0)
#         nan_std = np.nanstd(data_all, axis=0)
#         return nan_mean, nan_std
#
#     def _local_normalization(self, feature: np.ndarray, variable: str) -> np.ndarray:
#         if variable == 'inputs':
#             feature = (feature - self.x_mean) / self.x_std
#         elif variable == 'output':
#             feature = (feature - self.y_mean) / self.y_std
#         else:
#             raise RuntimeError(f"Unknown variable type {variable}")
#         return feature
#
#     def normalize_data(self):
#         # Normalize data
#         for idx, basin in enumerate(self.basins_list):
#             print(self.stage, "Normalizing %.4f" % (idx / len(self.basins_list)))
#             x = self.x_dict[basin]
#             y = self.y_dict[basin]
#             # Normalize data
#             x_norm = self._local_normalization(x, variable='inputs')
#             y_norm = self._local_normalization(y, variable='output')
#             self.x_dict[basin] = x_norm
#             self.y_dict[basin] = y_norm
#
#     def local_rescale(self, feature: np.ndarray, variable: str) -> np.ndarray:
#         if variable == 'inputs':
#             feature = feature * self.x_std + self.x_mean
#         elif variable == 'output':
#             feature = feature * self.y_std + self.y_mean
#         else:
#             raise RuntimeError(f"Unknown variable type {variable}")
#         return feature
#
#     def get_means(self):
#         return self.x_mean, self.y_mean
#
#     def get_stds(self):
#         return self.x_std, self.y_std
#
#     @classmethod
#     def get_instance(cls, past_len: int, pred_len: int, stage: str, specific_cfg: dict,
#                      x_mean=None, y_mean=None, x_std=None, y_std=None, y_stds_dict=None):
#         final_data_path = specific_cfg["final_data_path"]
#         camels_root = specific_cfg["camels_root"]
#         basins_list = specific_cfg["basins_list"]
#         forcing_type = specific_cfg["forcing_type"]
#         start_date = specific_cfg["start_date"]
#         end_date = specific_cfg["end_date"]
#         if final_data_path is None:
#             dates = [start_date, end_date]
#             instance = cls(camels_root, forcing_type, basins_list, past_len, pred_len, stage,
#                            dates, x_mean, y_mean, x_std, y_std, y_stds_dict)
#             return instance
#         else:
#             if final_data_path.exists():
#                 instance = torch.load(final_data_path)
#                 return instance
#             else:
#                 dates = [start_date, end_date]
#                 instance = cls(camels_root, forcing_type, basins_list, past_len, pred_len, stage,
#                                dates, x_mean, y_mean, x_std, y_std, y_stds_dict)
#                 final_data_path.parent.mkdir(exist_ok=True, parents=True)
#                 torch.save(instance, final_data_path)
#                 return instance


"""---------------"""
WINDOW_SIZE = 7
x_col = ['precipitation_AWAP', 'et_morton_actual_SILO',
         'tmax_awap', 'tmin_awap', 'vprp_awap']
y_col = ['streamflow_mmd']

# class CamelsAusDataset(object):
#     """Class to read Camels dataset from file
#     """
#
#     x_col = ['precipitation_AWAP', 'et_morton_actual_SILO',
#              'tmax_awap', 'tmin_awap', 'vprp_awap']
#     y_col = ['streamflow_mmd']
#     coord_col = ['station_id', 'time']
#
#     def __init__(self, data_dir, x_col=None, y_col=None,
#                  scale: bool = True, create_seq: bool = True,
#                  window_size: int = WINDOW_SIZE):
#
#         # Path to Camels data
#         self.data_dir = data_dir
#
#         # Create data repository
#         self.repo = CamelsAus()
#         self.repo.load_from_text_files(self.data_dir)
#
#         # Xarray dataset object
#         self.ds = self.repo.daily_data.sel(time=slice(dt.datetime(1980, 1, 1), dt.datetime(2015, 1, 1)))
#
#         # Define x and y columns
#         if x_col is not None:
#             self.x_col = x_col
#         if y_col is not None:
#             self.y_col = y_col
#
#         # DS list
#         self.ds_store = self.create_datasets(scale, create_seq, window_size=window_size)
#
#     def create_sequence(self, t, X, y, window_size):
#
#         assert window_size is not None, "Window size cannot be NoneType."
#
#         # Create empyty sequences
#         ts, Xs, ys = [], [], []
#
#         # Add sequences to Xs and ys
#         for i in range(len(X) - window_size):
#             Xs.append(X[i: (i + window_size)])
#             ys.append(y[i + window_size - 1])
#             ts.append(t[i + window_size - 1])
#
#         ts, Xs, ys = torch.stack(ts), torch.stack(Xs), torch.stack(ys)
#
#         return ts, Xs, ys
#
#     def create_datasets(self, scale, create_seq, window_size=None):
#         # Store station ids
#         self.stations = self.ds.station_id.to_numpy()
#
#         X_list, y_list, coord_list = [], [], []
#
#         for station_id in self.stations:
#             station_ds = self.ds.sel(station_id=station_id)
#             station_ds = station_ds[self.x_col + self.y_col].where(
#                 lambda x: x[self.y_col[0]].notnull(),
#                 drop=True
#             )
#             for x_col in self.x_col:
#                 station_ds = station_ds[self.x_col + self.y_col].where(
#                     lambda x: x[x_col].notnull(),
#                     drop=True
#                 )
#             station_df = station_ds.to_pandas().reset_index()
#
#             station_df.time = station_df.time.apply(lambda x: time.mktime(x.timetuple()))
#
#             X_list.append(station_df[self.x_col])
#             y_list.append(station_df[self.y_col])
#             coord_list.append(station_df[self.coord_col])
#
#         X = pd.concat(X_list, axis=0).reset_index(drop=True)
#         y = pd.concat(y_list, axis=0).reset_index(drop=True)
#         coord = pd.concat(coord_list, axis=0).reset_index(drop=True)
#
#         # Scaling preference
#         self.scale = scale
#         if scale:
#             self.x_scaler = StandardScaler()
#             self.y_scaler = StandardScaler()
#             # Scale
#             X = self.x_scaler.fit_transform(X)
#             y = self.y_scaler.fit_transform(y)
#
#         else:
#             X = X.values
#             y = y.values
#
#         ds_store = {}
#
#         for station_id in self.stations:
#
#             indices = coord.index[coord.station_id == station_id]
#
#             indices_train, indices_val = train_test_split(indices,
#                                                           test_size=0.3,
#                                                           shuffle=False)
#             X_train, X_val = (
#                 torch.from_numpy(X[indices_train]),
#                 torch.from_numpy(X[indices_val])
#             )
#             y_train, y_val = (
#                 torch.from_numpy(y[indices_train]),
#                 torch.from_numpy(y[indices_val])
#             )
#             time_train, time_val = (
#                 torch.from_numpy(
#                     coord.values[indices_train, 1].astype('float')
#                 ), torch.from_numpy(
#                     coord.values[indices_val, 1].astype('float')
#                 )
#             )
#
#             # Create Sequences
#             if create_seq:
#                 time_train, X_train, y_train = self.create_sequence(
#                     time_train, X_train, y_train,
#                     window_size=window_size
#                 )
#
#                 time_val, X_val, y_val = self.create_sequence(
#                     time_val, X_val, y_val,
#                     window_size=window_size
#                 )
#
#             ds_store[station_id] = {
#                 'train': data.TensorDataset(time_train, X_train, y_train),
#                 'val': data.TensorDataset(time_val, X_val, y_val)
#             }
#
#         return ds_store
#
#     def get_dataloader(self, station_id, train=True, batch_size=64, shuffle=False):
#
#         if train:
#             return data.DataLoader(
#                 self.ds_store[station_id]['train'], shuffle=shuffle, batch_size=batch_size
#             )
#         else:
#             return data.DataLoader(
#                 self.ds_store[station_id]['val'], shuffle=shuffle, batch_size=batch_size
#             )
"""------"""


class CamelsAusDataset2(Dataset):
    # x_col = ['precipitation_AWAP', 'et_morton_actual_SILO',
    #          'tmax_awap', 'tmin_awap', 'vprp_awap']
    # y_col = ['streamflow_mmd']
    # coord_col = ['station_id', 'time']

    def __init__(self, camels_root: str, basins_list: list, past_len: int, pred_len: int, stage: str,
                 dates: list, x_mean=None, y_mean=None, x_std=None, y_std=None, y_stds_dict=None):

        self.camels_root = camels_root  # 数据集目录
        self.basins_list = basins_list
        self.past_len = past_len
        self.pred_len = pred_len
        self.stage = stage
        self.dates = dates
        self.x_dict = dict()
        self.y_dict = dict()
        self.date_index_dict = dict()
        self.length_ls = list()

        self.repo = CamelsAus()
        # if len(self.basins_list) == 222:
        #     self.repo.load_from_text_files(self.camels_root, version="1.0")
        # else:
        #     self.repo.load_from_text_files(self.camels_root, version="2.0")
        self.repo.load_from_text_files(self.camels_root, version="2.0")
        self.ds = self.repo.daily_data.sel(time=slice(dates[0], dates[1]))

        if x_col is not None:
            self.x_col = x_col
        if y_col is not None:
            self.y_col = y_col
        if y_stds_dict is None:
            self.y_stds_dict = dict()
        else:
            self.y_stds_dict = y_stds_dict

        self._load_data()
        if self.stage == 'train':
            self.x_mean, self.x_std = self.calc_mean_and_std(self.x_dict)
            self.y_mean, self.y_std = self.calc_mean_and_std(self.y_dict)
        else:
            self.x_mean = x_mean
            self.y_mean = y_mean
            self.x_std = x_std
            self.y_std = y_std
        self.normalize_data()

        self.num_samples = 0
        for item in self.length_ls:
            self.num_samples += item

        self.index_ls = [0]
        for i in range(len(self.length_ls)):
            v = self.index_ls[i] + self.length_ls[i]
            self.index_ls.append(v)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        basin_idx = bisect_right(self.index_ls, idx) - 1
        local_idx = idx - self.index_ls[basin_idx]
        basin = self.basins_list[basin_idx]
        x_seq = self.x_dict[basin][local_idx: local_idx + self.past_len + self.pred_len, :]
        y_seq_past = self.y_dict[basin][local_idx: local_idx + self.past_len, :]
        y_seq_future = self.y_dict[basin][local_idx + self.past_len: local_idx + self.past_len + self.pred_len, :]

        return x_seq, y_seq_past, y_seq_future, self.y_stds_dict[basin]

    def _load_data(self):
        # Store station ids
        self.stations = self.ds.station_id.to_numpy()

        X_list, y_list, coord_list = [], [], []

        for station_id in self.stations:
            station_ds = self.ds.sel(station_id=station_id)
            station_ds = station_ds[self.x_col + self.y_col].where(
                lambda x: x[self.y_col[0]].notnull(),
                drop=True
            )  # 删除null值
            for x_col in self.x_col:
                station_ds = station_ds[self.x_col + self.y_col].where(
                    lambda x: x[x_col].notnull(),
                    drop=True
                )
            station_df = station_ds.to_pandas().reset_index()

            station_df.time = station_df.time.apply(lambda x: time.mktime(x.timetuple()))

            X_list.append(station_df[self.x_col])
            y_list.append(station_df[self.y_col])
            coord_list.append(station_df[self.coord_col])

        X = pd.concat(X_list, axis=0).reset_index(drop=True)
        y = pd.concat(y_list, axis=0).reset_index(drop=True)
        coord = pd.concat(coord_list, axis=0).reset_index(drop=True)

        for idx, station_id in enumerate(self.stations):
            print(self.stage, f"{station_id}: loading data %.4f" % (idx / len(self.stations)))
            indices = coord.index[coord.station_id == station_id]
            df_x = X.loc[indices].values.astype("float32")
            df_y = y.loc[indices].values.astype("float32")
            df_t = coord.values[indices, 1].astype('float')
            self.date_index_dict[station_id] = df_t

            self.x_dict[station_id] = df_x
            self.y_dict[station_id] = df_y

            self.length_ls.append(len(df_x) - self.past_len - self.pred_len + 1)
            if self.stage == 'train':
                self.y_stds_dict[station_id] = y.std(axis=0).item()

    @staticmethod
    def calc_mean_and_std(data_dict):
        data_all = np.concatenate(list(data_dict.values()), axis=0)  # CAN NOT serializable
        nan_mean = np.nanmean(data_all, axis=0)
        nan_std = np.nanstd(data_all, axis=0)
        return nan_mean, nan_std

    def _local_normalization(self, feature: np.ndarray, variable: str) -> np.ndarray:
        if variable == 'inputs':
            feature = (feature - self.x_mean) / self.x_std
        elif variable == 'output':
            feature = (feature - self.y_mean) / self.y_std
        else:
            raise RuntimeError(f"Unknown variable type {variable}")
        return feature

    def normalize_data(self):
        # Normalize data
        for idx, basin in enumerate(self.basins_list):
            print(self.stage, "Normalizing %.4f" % (idx / len(self.basins_list)))
            x = self.x_dict[basin]
            y = self.y_dict[basin]
            # Normalize data
            x_norm = self._local_normalization(x, variable='inputs')
            y_norm = self._local_normalization(y, variable='output')
            self.x_dict[basin] = x_norm
            self.y_dict[basin] = y_norm

    def local_rescale(self, feature: np.ndarray, variable: str) -> np.ndarray:
        if variable == 'inputs':
            feature = feature * self.x_std + self.x_mean
        elif variable == 'output':
            feature = feature * self.y_std + self.y_mean
        else:
            raise RuntimeError(f"Unknown variable type {variable}")
        return feature

    def get_means(self):
        return self.x_mean, self.y_mean

    def get_stds(self):
        return self.x_std, self.y_std

    @classmethod
    def get_instance(cls, past_len: int, pred_len: int, stage: str, specific_cfg: dict,
                     x_mean=None, y_mean=None, x_std=None, y_std=None, y_stds_dict=None):
        final_data_path = specific_cfg["final_data_path"]
        camels_root = specific_cfg["camels_root"]
        basins_list = specific_cfg["basins_list"]
        forcing_type = specific_cfg["forcing_type"]
        start_date = specific_cfg["start_date"]
        end_date = specific_cfg["end_date"]
        if final_data_path is None:
            dates = [start_date, end_date]
            instance = cls(camels_root, basins_list, past_len, pred_len, stage,
                           dates, x_mean, y_mean, x_std, y_std, y_stds_dict, )
            return instance
        else:
            if final_data_path.exists():
                instance = torch.load(final_data_path)
                return instance
            else:
                dates = [start_date, end_date]
                instance = cls(camels_root, basins_list, past_len, pred_len, stage,
                               dates, x_mean, y_mean, x_std, y_std, y_stds_dict)
                final_data_path.parent.mkdir(exist_ok=True, parents=True)
                torch.save(instance, final_data_path)  # 保存，方便下次使用
                return instance


class CamelsAusDatasetWithStatic(CamelsAusDataset2):
    def __init__(self, camels_root: str, basins_list: list, past_len: int, pred_len: int, stage: str,
                 dates: list, x_mean=None, y_mean=None, x_std=None, y_std=None, y_stds_dict=None):
        self.df_static = None
        self.static_file_path = None
        self.norm_static_fea = dict()
        super().__init__(camels_root, basins_list, past_len, pred_len, stage,
                         dates, x_mean, y_mean, x_std, y_std, y_stds_dict)

    def _load_data(self):
        # self.stations = self.ds.station_id.to_numpy()
        self.stations = self.basins_list
        X_list, y_list, coord_list = [], [], []
        for station_id in self.stations:

            station_ds = self.ds.sel(station_id=station_id)
            station_ds = station_ds[self.x_col + self.y_col].where(
                lambda x: x[self.y_col[0]].notnull(),
                drop=True
            )
            for x_col in self.x_col:
                station_ds = station_ds[self.x_col + self.y_col].where(
                    lambda x: x[x_col].notnull(),
                    drop=True
                )
            station_df = station_ds.to_pandas().reset_index()

            station_df.time = station_df.time.apply(lambda x: time.mktime(x.timetuple()))

            X_list.append(station_df[self.x_col])
            y_list.append(station_df[self.y_col])
            coord_list.append(station_df[self.coord_col])

        X = pd.concat(X_list, axis=0).reset_index(drop=True)
        y = pd.concat(y_list, axis=0).reset_index(drop=True)
        coord = pd.concat(coord_list, axis=0).reset_index(drop=True)

        for idx, station_id in enumerate(self.stations):
            print(self.stage, f"{station_id}: loading data %.4f" % (idx / len(self.stations)))
            indices = coord.index[coord.station_id == station_id]
            df_x = X.loc[indices].values.astype("float32")
            df_y = y.loc[indices].values.astype("float32")
            df_t = coord.values[indices, 1].astype('float')
            self.date_index_dict[station_id] = df_t

            self.x_dict[station_id] = df_x
            self.y_dict[station_id] = df_y
            self.norm_static_fea[station_id] = self.get_df_static(station_id)

            self.length_ls.append(len(df_x) - self.past_len - self.pred_len + 1)
            if self.stage == 'train':
                self.y_stds_dict[station_id] = y.std(axis=0).item()

    def normalize_data(self):
        # Normalize data
        for idx, basin in enumerate(self.basins_list):
            print(self.stage, "Normalizing %.4f" % (idx / len(self.basins_list)))
            x = self.x_dict[basin]
            y = self.y_dict[basin]
            # Normalize data
            x_norm = self._local_normalization(x, variable='inputs')
            y_norm = self._local_normalization(y, variable='output')
            # norm_static_fea = self.norm_static_fea[basin].repeat(x_norm.shape[0], axis=0)
            # x_norm_static = np.concatenate([x_norm, norm_static_fea], axis=1)
            self.x_dict[basin] = x_norm
            self.y_dict[basin] = y_norm

    def __getitem__(self, idx: int):
        basin_idx = bisect_right(self.index_ls, idx) - 1
        local_idx = idx - self.index_ls[basin_idx]
        basin = self.basins_list[basin_idx]
        x_seq = self.x_dict[basin][local_idx: local_idx + self.past_len + self.pred_len, :]
        y_seq_past = self.y_dict[basin][local_idx: local_idx + self.past_len, :]
        y_seq_future = self.y_dict[basin][local_idx + self.past_len: local_idx + self.past_len + self.pred_len, :]

        return x_seq, y_seq_past, y_seq_future, self.y_stds_dict[basin], self.norm_static_fea[basin]

    def get_df_static(self, basin):
        self.static_file_path = self.camels_root / "04_attributes" / "b.csv"
        self.df_static = pd.read_csv(self.static_file_path, header=0, dtype={"station_id": str}).set_index("station_id")
        return self.df_static.loc[[basin]].fillna(0).values


class CamelsAusDatasetWithStatic_decompose(CamelsAusDataset2):
    def __init__(self, camels_root: str, basins_list: list, past_len: int, pred_len: int, stage: str,
                 dates: list, x_mean=None, y_mean=None, x_std=None, y_std=None, y_stds_dict=None,
                 decompose=None, decompose_var="y"):
        self.df_static = None
        self.static_file_path = None
        self.decompose = decompose
        self.decompose_var = decompose_var
        if self.decompose == 'pca':
            self.decompose_var = "x"
        self.norm_static_fea = dict()
        super().__init__(camels_root, basins_list, past_len, pred_len, stage,
                         dates, x_mean, y_mean, x_std, y_std, y_stds_dict)

    def _load_data(self):
        # self.stations = self.ds.station_id.to_numpy()
        self.stations = self.basins_list
        X_list, y_list, coord_list = [], [], []
        for station_id in self.stations:
            station_ds = self.ds.sel(station_id=station_id)
            station_ds = station_ds[self.x_col + self.y_col].where(
                lambda x: x[self.y_col[0]].notnull(),
                drop=True
            )
            for x_col in self.x_col:
                station_ds = station_ds[self.x_col + self.y_col].where(
                    lambda x: x[x_col].notnull(),
                    drop=True
                )
            station_df = station_ds.to_pandas().reset_index()

            station_df.time = station_df.time.apply(lambda x: time.mktime(x.timetuple()))

            X_list.append(station_df[self.x_col])
            y_list.append(station_df[self.y_col])
            coord_list.append(station_df[self.coord_col])

        X = pd.concat(X_list, axis=0).reset_index(drop=True)
        y = pd.concat(y_list, axis=0).reset_index(drop=True)
        coord = pd.concat(coord_list, axis=0).reset_index(drop=True)

        for idx, station_id in enumerate(self.stations):
            print(self.stage, f"{station_id}: loading data %.4f" % (idx / len(self.stations)))
            indices = coord.index[coord.station_id == station_id]
            df_x = X.loc[indices].values.astype("float32")
            df_y = y.loc[indices].values.astype("float32")
            if self.decompose == "mvmd" and self.decompose_var == "y":
                y_copy = df_y.copy()
                df_y = torch.from_numpy(df_y).permute(1, 0)
                [u, u_hat, omega] = mvmd(df_y, 2000, 0, 3, 0, 1, 1e-7, 50)
                df_y = u.permute(2, 1, 0)[0].numpy()
                df_y = np.append(df_y, y_copy, axis=1)
            elif self.decompose == "memd" and self.decompose_var == "y":
                y_copy = df_y.copy()  # 分解前的y
                imfs = select_imfs_by_variance(df_y[:, 0], target_num=3)
                df_y = np.append(imfs.transpose(1, 0), y_copy, axis=1)
            elif self.decompose == "stl" and self.decompose_var == "y":
                y_copy = df_y.copy()  # 分解前的y
                stl = STL(df_y[:, 0], period=365, seasonal=9)
                result = stl.fit()
                df_y = np.vstack([result.seasonal, result.trend, result.resid]).transpose(1, 0)
                df_y = np.append(df_y, y_copy, axis=1)
            elif self.decompose == "ssa" and self.decompose_var == "y":
                y_copy = df_y.copy()  # 分解前的y
                ssa = SingularSpectrumAnalysis(window_size=365, groups=[[0, 1], [2, 3, 4, 5, 6], list(range(7, 365))])
                df_y_ssa = ssa.fit_transform(y.T)
                df_y = np.append(df_y_ssa[0].T, y_copy, axis=1)

            df_t = coord.values[indices, 1].astype('float')
            self.date_index_dict[station_id] = df_t
            self.x_dict[station_id] = df_x
            self.y_dict[station_id] = df_y
            self.norm_static_fea[station_id] = self.get_df_static(station_id)

            self.length_ls.append(len(df_x) - self.past_len - self.pred_len + 1)
            if self.stage == 'train':
                self.y_stds_dict[station_id] = y.std(axis=0)

    def normalize_data(self):
        # Normalize data
        for idx, basin in enumerate(self.basins_list):
            print(self.stage, "Normalizing %.4f" % (idx / len(self.basins_list)))
            x = self.x_dict[basin]
            y = self.y_dict[basin]
            # Normalize data
            x_norm = self._local_normalization(x, variable='inputs')
            y_norm = self._local_normalization(y, variable='output')
            # norm_static_fea = self.norm_static_fea[basin].repeat(x_norm.shape[0], axis=0)
            # x_norm_static = np.concatenate([x_norm, norm_static_fea], axis=1)

            if self.decompose == "pca":
                pca = PCA(n_components=3)
                x_norm = pca.fit_transform(x_norm)
                # df_y = np.append(imfs.transpose(1, 0), y_copy, axis=1)

            self.x_dict[basin] = x_norm
            self.y_dict[basin] = y_norm

    def local_rescale(self, feature: np.ndarray, variable: str) -> np.ndarray:
        if variable == 'inputs':
            feature = feature * self.x_std + self.x_mean
        elif variable == 'output':
            feature = feature * self.y_std[-1] + self.y_mean[-1]
        else:
            raise RuntimeError(f"Unknown variable type {variable}")
        return feature

    def __getitem__(self, idx: int):
        basin_idx = bisect_right(self.index_ls, idx) - 1
        local_idx = idx - self.index_ls[basin_idx]
        basin = self.basins_list[basin_idx]
        x_seq = self.x_dict[basin][local_idx: local_idx + self.past_len + self.pred_len, :]
        if self.decompose_var == 'y':
            y_seq_past = self.y_dict[basin][local_idx: local_idx + self.past_len, :-1]
            y_seq_future = self.y_dict[basin][local_idx + self.past_len: local_idx + self.past_len + self.pred_len, -1:]
        else:
            y_seq_past = self.y_dict[basin][local_idx: local_idx + self.past_len, :]
            y_seq_future = self.y_dict[basin][local_idx + self.past_len: local_idx + self.past_len + self.pred_len, :]

        return x_seq, y_seq_past, y_seq_future, self.y_stds_dict[basin].values.astype("float32"), self.norm_static_fea[
            basin]

    def get_df_static(self, basin):
        self.static_file_path = self.camels_root / "04_attributes" / "b.csv"
        self.df_static = pd.read_csv(self.static_file_path, header=0, dtype={"station_id": str}).set_index("station_id")
        return self.df_static.loc[[basin]].fillna(0).values

    @classmethod
    def get_instance(cls, past_len: int, pred_len: int, stage: str, specific_cfg: dict,
                     x_mean=None, y_mean=None, x_std=None, y_std=None, y_stds_dict=None):
        final_data_path = specific_cfg["final_data_path"]
        camels_root = specific_cfg["camels_root"]
        basins_list = specific_cfg["basins_list"]
        start_date = specific_cfg["start_date"]
        end_date = specific_cfg["end_date"]
        decompose = specific_cfg["decompose"]
        if final_data_path is None:
            dates = [start_date, end_date]
            instance = cls(camels_root, basins_list, past_len, pred_len, stage,
                           dates, x_mean, y_mean, x_std, y_std, y_stds_dict, decompose=decompose)
            return instance
        else:
            if final_data_path.exists():
                instance = torch.load(final_data_path)
                return instance
            else:
                dates = [start_date, end_date]
                instance = cls(camels_root, basins_list, past_len, pred_len, stage,
                               dates, x_mean, y_mean, x_std, y_std, y_stds_dict, decompose=decompose)
                final_data_path.parent.mkdir(exist_ok=True, parents=True)
                torch.save(instance, final_data_path)  # 保存，方便下次使用
                return instance


if __name__ == '__main__':
    from pathlib import Path

    # data_path = r'D:\RR-Former\camels_aus'
    # ds = CamelsAusDataset(data_dir=data_path)

    train_start = pd.to_datetime("1980-10-01", format="%Y-%m-%d")
    train_end = pd.to_datetime("1995-09-30", format="%Y-%m-%d")
    test_start = pd.to_datetime("2000-10-01", format="%Y-%m-%d")
    test_end = pd.to_datetime("2014-09-30", format="%Y-%m-%d")
    dates = [train_start, train_end]

    camels_root = Path(r"D:\RR-Former\camels_aus")
    basins_file = r"D:\RR-Former\camels_aus\aus_561basins_list.txt"
    # camels_root = Path(r"D:\ealstm\CAMELS_US")
    # basins_file = r"D:\RR-Former\data\448basins_list.txt"

    global_basins_list = pd.read_csv(basins_file, header=None, dtype=str)[0].values.tolist()
    past_len = 365
    pred_len = 7
    stage = "test"
    # ds = CamelsAusDatasetWithStatic_decompose(camels_root=camels_root, basins_list=global_basins_list[:1],
    #                                           past_len=past_len, pred_len=pred_len, stage=stage,
    #                                           dates=dates, decompose="memd", decompose_var="y")
    ds = CamelsAusDatasetWithStatic_decompose(camels_root=camels_root, basins_list=["803003", "804001"],
                                              past_len=past_len, pred_len=pred_len, stage=stage,
                                              dates=dates, decompose="stl", decompose_var="y",
                                              x_mean=np.random.rand(5), x_std=np.random.rand(5),
                                              y_mean=np.random.rand(4), y_std=np.random.rand(4))
    x_seq, y_seq_past, y_seq_future, y_stds, static_fea = ds[0]
    print(len(ds))
