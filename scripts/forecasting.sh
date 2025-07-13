python -u forecast_train.py CaTT ETTh1 forecast_univar --loader forecast_csv_univar --repr-dims 320 --max-threads 8 --seed 1 --eval
python -u forecast_train.py CaTT ETTh2 forecast_univar --loader forecast_csv_univar --repr-dims 320 --max-threads 8 --seed 1 --eval
python -u forecast_train.py CaTT ETTm1 forecast_univar --loader forecast_csv_univar --repr-dims 320 --max-threads 8 --seed 1 --eval
python -u forecast_train.py CaTT weather forecast_univar --loader forecast_csv_univar --repr-dims 320 --max-threads 8 --seed 1 --eval