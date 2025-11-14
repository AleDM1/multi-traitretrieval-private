from .model_module_F import samp_w
from .feature_module_F import plot_fig

from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score 

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
from pickle import dump,load
from tensorflow.keras.models import model_from_json

from scipy import stats

import shap


def collectPredictions(paths, name=''):
    data = pd.DataFrame()

    for dir_n in paths:
        sc = pd.read_csv(dir_n + '{}.csv'.format(name)).drop(['Unnamed: 0'],axis=1)
        data = pd.concat([data,sc],axis=0)

    return data.reset_index(drop=True)


def load_model(dir_data, gp = None):
    if(gp is not None):
        json_file = open(dir_data + 'Model_db{}.json'.format(gp), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        best_model = model_from_json(loaded_model_json)

        scaler_list = load(open(dir_data + 'scaler_db{}.pkl'.format(gp), 'rb'))

        # load weights into new model
        best_model.load_weights(dir_data + 'Trial_db{}_weights.h5'.format(gp))
    else:
        json_file = open(dir_data + 'Model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        best_model = model_from_json(loaded_model_json)

        scaler_list = load(open(dir_data+ 'scaler_global.pkl', 'rb'))

        # load weights into new model
        best_model.load_weights(dir_data+ 'Trial_weights.h5')
    
    return best_model, scaler_list

def all_scores(test_tr,Traits,obs_pf, pred_df,samp_w_ts=None, method = None, save =False, dir_n = None):
    r2_tab = []
    RMSE_tab = []
    nrmse_tab = []
    mae_tab = []
    b_tab = []

    for j in test_tr:

        f = pred_df[j+ ' Predictions'].reset_index(drop=True) # + ' Predictions'
        y = obs_pf[j].reset_index(drop=True)

        idx = np.union1d(f[f.isna()].index,y[y.isna()].index)

        f.drop(idx, axis = 0, inplace=True)
        y.drop(idx, axis = 0, inplace=True)
        
        
        if (y.notnull().sum()):
            if (samp_w_ts is not None):
                we = pd.DataFrame(samp_w_ts).loc[f.index,:]
            else:
                we = None

            if (we is not None) and (we.sum().sum() !=0):
                r2_tab.append(r2_score(y,f,sample_weight= we))

                RMSE=math.sqrt(mean_squared_error(y,f,sample_weight= we))
                RMSE_tab.append(RMSE)
                nrmse_tab.append((RMSE*100)/(np.nanquantile(np.array(y),0.99) - np.nanquantile(np.array(y),0.01)))

                mae_tab.append(mean_absolute_error(y,f,sample_weight= we))

                bias=np.sum(np.array(y)-np.array(f))/len(f)
                b_tab.append(bias)
            else:
                r2_tab.append(r2_score(y,f))

                RMSE=math.sqrt(mean_squared_error(y,f))
                RMSE_tab.append(RMSE)
                nrmse_tab.append((RMSE*100)/(np.nanquantile(np.array(y),0.99) - np.nanquantile(np.array(y),0.01)))

                mae_tab.append(mean_absolute_error(y,f))

                bias=np.sum(np.array(y)-np.array(f))/len(f)
                b_tab.append(bias)
        else:
            r2_tab.append(np.nan)
            RMSE_tab.append(np.nan)
            nrmse_tab.append(np.nan)
            mae_tab.append(np.nan)
            b_tab.append(np.nan)
            pass        

    test = pd.DataFrame([r2_tab, RMSE_tab, nrmse_tab,mae_tab,b_tab], columns= test_tr[:len(test_tr)], index=['r2_score','RMSE','nRMSE (%)','MAE','Bias'])
    if(save):
        test.to_csv(dir_n + 'scores_all_{}.csv'.format(method))
    return test


def scatterPlot(obs, preds, Traits, test_tr, test, meta=None, sp = None, method = None, save =False, dir_n = None, figsize=(10, 13), quantile_vis=0.999, size=5.5):

    if sp is not None:
        n = len(meta[sp].unique())
        cmap = sns.color_palette('colorblind', n)

    plt.rc('font', size=size) #5.7
    plt.rcParams['lines.markersize'] = 4
    plt.rcParams['lines.linewidth'] = 0.5

    a = round((len(test_tr))/4) # number of rows
    b = 4  # number of columns
    c = 1  # initialize plot counter

    fig = plt.figure(figsize = figsize,dpi=300 ,constrained_layout=True)

    for j in range(len(test_tr)):
        f = preds.loc[:,Traits[Traits.index(test_tr[j])]+' Predictions']
        y = obs.loc[:,Traits[Traits.index(test_tr[j])]]

        idx = np.union1d(f[f.isna()].index,y[y.isna()].index)

        f.drop(idx, axis = 0, inplace=True)
        y.drop(idx, axis = 0, inplace=True)
        
        if meta is not None:
            m = meta.drop(idx, axis = 0)

        ax1 = plt.subplot(a, b, c)
        plt.axis('square')
        
        
        lim_max = min(f.quantile(quantile_vis),y.quantile(quantile_vis)) 
        lim_min = max(f.quantile(1-quantile_vis),y.quantile(1-quantile_vis))
        
        #######
        ax1.set_xlim(lim_min,lim_max)
        ax1.set_ylim(ax1.get_xlim())
        ax1.set_aspect('equal', adjustable='box')
        
        sns.lineplot(x=(lim_min,lim_max), y=(lim_min,lim_max), ax = ax1, color='black',legend='full', linestyle='dashed')
        
        y.name = y.name +' Observations'
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(f,y)
        sns.regplot(x= f, y=y, color='b', fit_reg= True, ax=ax1, scatter=False)

        if meta is not None:
            groups = pd.concat([f,y,m], axis=1).groupby(sp)

            for name, group in groups:
                ax1 = sns.regplot(x = group[test_tr[j]+' Predictions'], y = group[test_tr[j]+' Observations'] ,ax = ax1, fit_reg= False, ci=False, label= sp + ' {}'.format(group[sp].unique()[0]), scatter_kws={ "color": cmap[list(meta[sp].unique()).index(name)], 'alpha':0.6})
                ax1.set(xticks=ax1.get_xticks(), yticks=ax1.get_xticks())

        else:
            ax1 = sns.regplot(x = f, y = y ,fit_reg= False, ci=False, ax = ax1, scatter_kws={"color": "blue", 'alpha':0.6})
            ax1.set(xticks=ax1.get_xticks(), yticks=ax1.get_xticks())
        
        ann = 'y = {0:.2f}x+{1:.2f} \n R² = {2:.2f} \n nRMSE = {3:.2f}'.format(slope,intercept,test.loc['r2_score',test_tr][j],test.loc['nRMSE (%)',test_tr][j])

        ax1.annotate(ann,
            xy=(0.5,0.01),
            xycoords='axes fraction',
            horizontalalignment='left',
            verticalalignment='bottom',size=size)
        
        ann = test_tr[j]
        ax1.set_title(ann, y=1.1, pad=-5, fontdict = {'fontsize':size,
        'horizontalalignment': 'center', 'fontweight':'bold'})
        
        plt.xlabel(" ")
        plt.ylabel(" ")

        c = c + 1

    fig.supxlabel('Predictions', size=8, fontweight='bold', ha='center')
    fig.supylabel('Observations', size=8, fontweight='bold', ha='center')

    if(save):
        plt.savefig(dir_n + "ScatterFill_{}.pdf".format(method),bbox_inches = 'tight', dpi = 300)
        plt.savefig(dir_n + "ScatterFill_{}.svg".format(method),bbox_inches = 'tight', dpi = 300)
        plt.savefig(dir_n + "ScatterFill_{}.png".format(method),bbox_inches = 'tight', dpi = 300)

        
def histPlot(obs, preds, Traits, test_tr, method = None, save =False, dir_n = None,figsize=(15,40)):
    
    plt.rc('font', size=10)
    plt.rcParams['lines.markersize'] = 1
    plt.rcParams['lines.linewidth'] = 0.5
    
    a = round((len(test_tr)+1)/2)  # number of rows
    b = 2  # number of columns
    c = 1  # initialize plot counter

    fig = plt.figure(figsize=figsize)

    for j in range(len(test_tr)):

        f = preds.loc[:,Traits[Traits.index(test_tr[j])]+' Predictions']
        y = obs.loc[:,Traits[Traits.index(test_tr[j])]]

        idx = np.union1d(f[f.isna()].index,y[y.isna()].index)

        f.drop(idx, axis = 0, inplace=True)
        y.drop(idx, axis = 0, inplace=True) 

        test = pd.DataFrame(np.array([y,f]).T , columns=['Observations','Predictions']) 

        plt.subplot(a, b, c)
        plt.xlabel(test_tr[j])
        sns.histplot(test, alpha=0.3 ,element="step", label = test_tr[j],stat="probability")

        c = c + 1
        
    if(save):
        plt.savefig(dir_n + "Traits_hist_plot_allDataset_{}.pdf".format(method),bbox_inches = 'tight', dpi = 600)
        plt.savefig(dir_n + "Traits_hist_plot_allDataset_{}.svg".format(method),bbox_inches = 'tight', dpi = 600)

    
def cleanana(score):
    r = score
    r[r<0] = 0
    r[r>1] = 5
    return r


def shap_plot(test_tr,Traits, glob, coef, columns, method = None, save =False, dir_n = None, figsize=(21,22)):
    plt.rc('font', size=8)
    plt.rcParams['lines.linewidth'] = 1

    a = round((len(test_tr))/3)  # number of rows
    b = 3  # number of columns
    c = 1  # initialize plot counter

    fig = plt.figure(figsize=figsize)

    # db, test_X, y = read_db('dataset/testCV_{}.csv'.format(1),sp=True)

    for j in range(len(test_tr)):
        plt.subplot(a, b, c)

        plt.title(test_tr[j])

        avg = np.abs(np.median([coef],axis=0))
        sc = (avg-avg.min())/(avg.max()-avg.min())
        plt.plot(columns,sc, color='blue')
        ####################

        avg = np.median(np.array(glob),axis=0)
        mean_shap = pd.DataFrame(avg).mean()

        av_tr = pd.DataFrame(avg).T[Traits.index(test_tr[j])] / mean_shap
        av_tr = av_tr.rolling(20).median()
        sc = (av_tr - av_tr.min())/(av_tr.max() - av_tr.min())

        plt.plot(columns, sc, color='black')

        c = c + 1

    # fig.supxlabel('Wavelengths (nm)', size=7, fontweight='bold', ha='center')
    # fig.supylabel('Relative feature importance', size=7, fontweight='bold', ha='center')
    if(save):
        fig.savefig(dir_n +'PlsrVSIncomp_feature_importance_all.pdf',bbox_inches = 'tight', dpi = 300)
        fig.savefig(dir_n +'PlsrVSIncomp_feature_importance_all.svg',bbox_inches = 'tight', dpi = 300)    
        fig.savefig(dir_n +'PlsrVSIncomp_feature_importance_all.png',bbox_inches = 'tight', dpi = 300)  
    

def scatter_indiv(obs_tr, pred_tr, a, b, c, quantile_vis):
    plt.subplot(a, b, c)
    plt.axis('square')
    
    lim_max = min(pred_tr.quantile(quantile_vis),obs_tr.quantile(quantile_vis))
    lim_min = max(pred_tr.quantile(1+-quantile_vis),obs_tr.quantile(1-quantile_vis))
    
    ax1 = sns.lineplot(x=(lim_min,lim_max), y=(lim_min,lim_max), color='black',legend='full', linestyle='dashed')          
    ax1.set_xlim(lim_min,lim_max)
    ax1.set_ylim(ax1.get_xlim())
    
    ax1 = sns.regplot(x= pred_tr, color='b', y=obs_tr,fit_reg= True ,ax=ax1,scatter_kws={'alpha':0.2})
    ax1.set(xticks=ax1.get_xticks(), yticks=ax1.get_xticks())

    plt.xlabel('')
    plt.ylabel('')
    
    return ax1