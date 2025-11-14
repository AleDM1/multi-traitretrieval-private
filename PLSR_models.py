# import os
import sys
from pickle import dump,load

from sklearn.preprocessing import RobustScaler,MinMaxScaler,PowerTransformer,StandardScaler

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score 

from sklearn.compose import TransformedTargetRegressor

from sklearn.cross_decomposition import PLSRegression  


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sb
import pandas as pd
import numpy as np
import math


from data_module_F import *
from feature_module_F import *
from model_module_F import balanceData,data_prep,create_path

import argparse



def eval_model(y_pred, test_y, samp_w_test=None):
    f = pd.DataFrame(y_pred)
    y = test_y.reset_index(drop=True).drop(f[f[0].isna()].index, axis=0)

    if (samp_w_test is not None):
        we = pd.DataFrame(samp_w_test).reset_index(drop=True).drop(f[f[0].isna()].index, axis=0)
    else:
        we = None

    f.dropna(inplace=True)

    if (we is not None) and (we.sum().sum() !=0):
        r2 = r2_score(y,f,sample_weight= we)
        RMSE = math.sqrt(mean_squared_error(y,f,sample_weight= we))
        nRMSE = (RMSE*100)/(np.nanquantile(np.array(y),0.99) - np.nanquantile(np.array(y),0.01))
        mae = mean_absolute_error(y,f,sample_weight= we)
        bias = np.sum(np.array(y)-np.array(f))/len(f)

    else:
        r2 = r2_score(y,f)
        RMSE = math.sqrt(mean_squared_error(y,f))
        nRMSE = (RMSE*100)/(np.nanquantile(np.array(y),0.99) - np.nanquantile(np.array(y),0.01))
        mae = mean_absolute_error(y,f)
        bias = np.sum(np.array(y)-np.array(f))/len(f)

    return r2, RMSE, nRMSE, mae, bias


def optimise_pls_cv(X, y, n_comp):               
    mse = []
    component = [i for i in range(1, n_comp)]        
    for i in component:  
        #print(i)
        pls = PLSRegression(n_components=i)  
        y_cv = cross_val_predict(pls, X, y, cv = 10)
        mse.append(mean_squared_error(y, y_cv))
        
        comp = 100*(i+1)/40          
       
        # Calculate and print the position of minimum in MSE      
    msemin = np.argmin(mse)      
    print("Suggested number of components: ", msemin+1)      
    
    # Define PLS object with optimal number of components      
    #pls_opt = PLSRegression(n_components=msemin+1).fit(X, y) 
    pls_opt = TransformedTargetRegressor(regressor = PLSRegression(n_components= msemin+1), transformer= PowerTransformer(method='box-cox')).fit(X, y) 

    y_c = pls_opt.predict(X)        
    # Cross-validation      
    y_cv = cross_val_predict(pls_opt, X, y, cv=10)        
    # Calculate scores for calibration and cross-validation
    
    f_c = pd.DataFrame(y_c)
    f_cv = pd.DataFrame(y_cv)
    f = pd.DataFrame(y)

    f_cv.drop(f_c[f_c[f_c.columns[0]].isna()].index, axis = 0, inplace = True)
    f.drop(f_c[f_c[f_c.columns[0]].isna()].index, axis = 0, inplace = True)

    f_c.drop(f_cv[f_cv[f_cv.columns[0]].isna()].index, axis = 0, inplace = True)
    f.drop(f_cv[f_cv[f_cv.columns[0]].isna()].index, axis = 0, inplace = True)

    f_c.dropna(inplace=True)
    f_cv.dropna(inplace=True)
    
    score_c = r2_score(f, f_c)      
    score_cv = r2_score(f, f_cv)        
    # Calculate mean squared error for calibration and cross validation      
    mse_c = mean_squared_error(f, f_c)      
    mse_cv = mean_squared_error(f, f_cv) 
    
    backup = sys.stdout
    sys.stdout = open(dir_n + "/PLS_Cv.txt", "a")
    print("Suggested number of components: ", msemin+1)

    print('R2 calib: %5.3f \n'  % score_c)      
    print('R2 CV: %5.3f \n'  % score_cv)      
    print('MSE calib: %5.3f \n' % mse_c)      
    print('MSE CV: %5.3f \n' % mse_cv) 

    sys.stdout = backup
    
    return pls_opt


def predictions(test_tr, Traits, features_test, best_model, j):
    pred_test = best_model.predict(features_test)
    f = pd.DataFrame(pred_test, columns= [test_tr[j]+' Predictions'])
    return f


# Create the parser
my_parser = argparse.ArgumentParser(description='Multi-trait modelling Process')

# Add the arguments
my_parser.add_argument('--route',
                       metavar='route',
                       type=str,
                       help='Path for experiment directory')

my_parser.add_argument('--path',
                       metavar='path',
                       type=str,
                       help='the path to data')


my_parser.add_argument('--seed',
                       metavar='seed',
                       type=int,
                       help='Seed for data splitting')

my_parser.add_argument('--epochs',
                       metavar='epochs',
                       type=int,default=300,
                       help='Training epochs')

my_parser.add_argument('--indices',
                       metavar='indices',
                       type=int,default=1,
                       help='Number of repetition')

my_parser.add_argument('--exp',
                       metavar='exp',
                       type=str,
                       help='Experiment name')


my_parser.add_argument('--kind',
                       metavar='kind',
                       type=str, default=None,
                       help='Model definition')

my_parser.add_argument('--lr',
                       metavar='lr',
                       type=float, default=0.0005,
                       help='Learning rate')

# Execute the parse_args() method
args = my_parser.parse_args()


path = args.path ## data path
seed = args.seed
exp = args.exp ## experiment name 
route = args.route


dir_n = route + '{}_{}/'.format(exp,seed) ## experiment dir 
create_path(dir_n)


if __name__ == "__main__":
    db_train, X_train, y_train = read_db(path + 'fillCV_{}.csv'.format(seed),sp=True)
    db_test, X_test, y_test = read_db(path + 'testCV_{}.csv'.format(seed),sp=True)

    fill = db_train.copy()

    samp_w_tr = pd.read_csv(path + 'samp_w_tr_{}.csv'.format(seed)).drop(['Unnamed: 0'],axis=1).loc[:,'0']

    r2_score_pls= []
    rmse_pls= []
    nrmse_pls= []
    mae_pls= []
    b_pls= []
    y_pred_pls = []

    for tr in range(len(Traits)):
        print(tr)
        train_x, train_y = data_prep('400', fill, Traits, i=tr)
        test_x, test_y = data_prep('400', db_test, Traits, i=tr)

        train_x = train_x.values
        train_y = train_y.values

        pls_opt= optimise_pls_cv(train_x, train_y, 20)
        dump(pls_opt, open(dir_n + '/model_{}.pkl'.format(tr), 'wb')) 

        y_pred = pls_opt.predict(test_x)
        y_pred_pls.append(y_pred)

        r2, RMSE, nRMSE, mae, b = eval_model(y_pred, test_y)
        r2_score_pls.append(r2)
        rmse_pls.append(RMSE)
        nrmse_pls.append(nRMSE)
        mae_pls.append(mae)
        b_pls.append(b)


    test_pls = pd.DataFrame(np.array([r2_score_pls, rmse_pls,nrmse_pls, mae_pls, b_pls]), index=['r2_score', 'RMSE', 'nRMSE (%)', 'MAE', 'Bias'], columns= Traits[:len(r2_score_pls)]).T
    test_pls.to_csv('./evaluation/PLS_scores_allTraits_{}.csv'.format(1))

    preds = []
    for j in range(len(Traits)):
        model = load(open(dir_n + '/model_{}.pkl'.format(Traits.index(Traits[j])), 'rb'))
        f = predictions(Traits, Traits, X_test, model, j)
        preds.append(f[Traits[j]+' Predictions'])

    pd.DataFrame(preds).T.to_csv('./Predictions/Plsr_predictions.csv')