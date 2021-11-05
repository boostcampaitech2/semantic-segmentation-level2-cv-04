# Semantic Segmentation for Recycling Trash

#### Rachel's Baseline Code

## How to use

1. Run `pip install -r requirements.txt`
2. Modify modules under `framework` as you experiment
3. Modify `train.sh` with args parameters you want to change (Refer to `utils/arg_parser.py` for parameters)
4. Run `bash train.sh`
5. After training, a model, yaml file, and log would be saved under `saved/exp_name`
6. Run `bash inference.sh` with a path of the yaml file you want to test
7. After inference, a submission csv file should be saved under `submission`
8. If you would like to visualize the csv file, please use `submission/CSV_visualize.ipynb` (for multiple csv files, use `submission/CSV_compare.ipynb`)