from  data_class import mydata
import asyncio
from modeles import *

async def get_coin_hist():
    md = mydata()
    await md.load_save_one_year_smb('DOGE', 5)

def test():
    md = mydata()
    df = md.load_data_from_local('data/DOGE5.json')
    data = md.take_data(df)
    #print(data)
    #X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test = split_scale_data(data)
    #start_r(data)
    #eval_2(data)
    #eval_3(data)
    eval_futures(data)


if __name__ == "__main__":
    test()

