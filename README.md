# Frequency domain convolutional network with historical data fusion module for Regional Streamflow Prediction

## Getting Started
This repository contains the code and resources related to the paper titled "Frequency_domain_convolutional_network_with_historical_data_fusion_module_for_Regional_Streamflow_Prediction".

1. Select your dataset and configure its path in **/configs/dataset_config.py**.
2. (Optional) Set the input sequence length in **/configs/data_shape_config.py**.
3. (Optional) Configure training parameters in **/run_config/pretrain_config.py**.
4. Run **pretrain.py** to pretrain the model.
5. Test the pretrained model globally using **pretrain_test_global.py**.

## Data Download

Download data from the following:
- [CAMELS-AUS V2](https://zenodo.org/records/14289037)
- [CAMELS](https://gdex.ucar.edu/dataset/camels.html)
