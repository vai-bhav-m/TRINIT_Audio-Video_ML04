import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import datetime

def Final_model_predict(given_date):
    ibm = pd.read_csv("daily_IBM.csv")
    ibm.fillna(0, inplace=True)

    ibm.drop(['open', 'high', 'low', 'volume'], axis=1, inplace=True)
    ibm.index = pd.to_datetime(ibm['timestamp'], format='%Y-%m-%d')
    ibm_mod = ibm.drop(['timestamp'], axis=1, inplace=False)
    date = "2013-01-02"
    current_date = datetime.datetime.strptime(given_date, "%Y-%m-%d")
    current_date -= datetime.timedelta(days=10)
    specified_date = current_date.strftime("%Y-%m-%d")
    truncated_1 = ibm_mod[ibm_mod.index > pd.to_datetime(date, format='%Y-%m-%d')]
    truncated = truncated_1[truncated_1.index < pd.to_datetime(specified_date, format='%Y-%m-%d')]
    test_set = truncated_1[truncated_1.index > pd.to_datetime(specified_date, format='%Y-%m-%d')]
    y = truncated['close'].values
    train_data = [x for x in y]
    test_data = test_set['close'].values

    model_predictions = []
    N_test_observations = len(test_data)

    for time_point in range(N_test_observations):
        model = ARIMA(train_data, order=(4, 1, 0))
        ARIMAmodel = model.fit()
        output = ARIMAmodel.forecast()
        yhat = output[0]
        model_predictions.append(yhat)
        true_test_value = test_data[time_point]
        train_data.append(true_test_value)
    Result = float(model_predictions[-1])
    MAE = abs(model_predictions[-1] - test_data[-1])
    #print('Testing Mean Squared Error is {}'.format(MSE_error))

    return Result, MAE

# k, l = Final_model_predict("2020-01-22")
# print(k, l)