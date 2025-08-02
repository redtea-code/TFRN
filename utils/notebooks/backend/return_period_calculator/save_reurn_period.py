import pandas as pd
import os

from backend.return_period_metrics import calculate_return_period_performance_metrics
from tqdm import tqdm

def save_period():
    data_path = r'E:\cyh\global_streamflow_model_paper-main\global_streamflow_model_paper-main\Model\result'
    # data_path = r'E:\cyh\ealstm_regional_modeling-master\ealstm_regional_modeling-master\results'#camel_us
    all_dt = pd.DataFrame([])
    for file in tqdm(os.listdir(data_path)):
        if file.endswith(".csv") and not file == 'return_period.csv':
            data_file = os.path.join(data_path, file)
            basin = file.split(".")[0]

            hydro_series = pd.read_csv(data_file, index_col=0)

            hydro_series.columns = ['qobs', 'qsim', 'month', 'day',"weekend", 'year']

            time_df = hydro_series[['month', 'day', 'year']]
            indx = pd.to_datetime(time_df)
            hydro_series.index = indx.astype("str")

            obs_series = hydro_series.iloc[:, 0]
            sim_seres = hydro_series.iloc[:, 1]

            try:
                dt = calculate_return_period_performance_metrics(observations=obs_series, predictions=sim_seres,
                                                                 temporal_resolution="1D")
            except:
                continue
            df1 = pd.DataFrame(dt[0], index=[basin])
            df2 = pd.DataFrame(dt[1], index=[basin])

            df = pd.concat([df1, df2], axis=1)
            all_dt = pd.concat([all_dt, df])
    print(all_dt.sum()/len(all_dt))
    save_path = os.path.join(data_path, "return_period.csv")
    if os.path.exists(save_path):
        os.remove(save_path)
        all_dt.to_csv(save_path)
    else:
        all_dt.to_csv(save_path)

if __name__ == '__main__':
    save_period()