import argparse

from data_module_F import *
from model_module_F import *
from feature_module_F import *
from evaluation_module_F import *


# from Resnet1D_builder import *
from EfficientNet1D_builder import *
from model_builder import *

import sys, os

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


my_parser.add_argument('--exp',
                       metavar='exp',
                       type=str,
                       help='Experiment name')

my_parser.add_argument('--gp',
                       metavar='gp',
                       type=bool, default=False,
                       help='With gap filling or not')

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
exp = args.exp ## experiment name 
seed = args.seed
GP = args.gp
route = args.route
epochs = args.epochs
lr = args.lr
kind = args.kind

dir_n = route + '{}_{}/'.format(exp,seed) ## experiment dir 
create_path(dir_n)

tf.random.set_seed(155)
    
######## GPU RAM memory ##########
os.environ["CUDA_VISIBLE_DEVICES"]="0"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 15*1GB of memory on the first GPU
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024*30)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
            print(e)

            
if __name__ == "__main__":

    #### Load data 
    db_train, X_train, y_train = read_db(path + 'fillCV_{}.csv'.format(seed),sp=True)
    db_test, X_test, y_test = read_db(path + 'testCV_{}.csv'.format(seed),sp=True)

    gap_fil = db_train.copy()

    samp_w_tr = pd.read_csv(path + 'samp_w_tr_{}.csv'.format(seed)).drop(['Unnamed: 0'],axis=1).loc[:,'0']

    
    #############Incomplete ############   

    path_g = dir_n + 'incomplete/'
    create_path(path_g)

    ######### Train set ######

    train_x, train_y= data_prep('400', gap_fil, Traits, multi= True)

        ##############
    scaler_list = save_scaler(train_y, save=True, dir_n = path_g, k = 'global')


    val_x = train_x.sample(frac = 0.1,random_state = seed)
    val_y = train_y.loc[val_x.index,:]
    samp_w_val = pd.DataFrame(samp_w_tr).sample(frac = 0.1,random_state = seed)

    if(samp_w_tr is not None):
        if (samp_w_tr.sum().sum() !=0):    
            train_ds = dataset(train_x.drop(val_x.index), train_y.drop(val_y.index), pd.DataFrame(samp_w_tr).drop(samp_w_val.index), scaler_list, Traits, shuffle=True,augment=True)
            test_ds = dataset(val_x, val_y, samp_w_val, scaler_list, Traits)
    else:
        train_ds = dataset(train_x.drop(val_x.index), train_y.drop(val_y.index), None, scaler_list, Traits, shuffle=True,augment=True)
        test_ds = dataset(val_x, val_y, None, scaler_list, Traits)

    ##### Model definition  and taraining #######
    input_shape = train_x.shape[1]
    output_shape = train_y.shape[1]

    EPOCHS = epochs 
    best_model = model_definition(input_shape, output_shape,lr = lr, kind= kind)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,restore_best_weights=True)

    checkpoint_path = path_g + 'checkpoint'
    create_path(checkpoint_path)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_path + "/epoch{epoch:02d}-val_root_mean_squared_error{val_root_mean_squared_error:.2f}.hdf5",
    save_weights_only=True,
    monitor = 'val_root_mean_squared_error',
    mode='min',
    save_best_only=True)

    his = best_model.fit(train_ds,
                    validation_data = test_ds,
                    epochs = EPOCHS,
                    verbose=1, callbacks = [model_checkpoint_callback])

    val_acc_per_epoch = his.history['val_root_mean_squared_error']
    best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1    

    path_trial = path_g + "Model.json"
    path_best = checkpoint_path + "/epoch{0:02d}-val_root_mean_squared_error{1:.2f}.hdf5".format(best_epoch,min(val_acc_per_epoch))
    path_w = path_g + 'Trial_weights.h5'
    save_model(best_model, path_trial, path_best, path_w)

    ########### Evaluation #############
    test_x, test_y = data_prep('400', db_test, Traits, multi = True)

    pred = scaler_list.inverse_transform(best_model.predict(test_x))
    pred_df = pd.DataFrame(pred, columns = test_y.columns+ ' Predictions')
    pred_df.to_csv(path_g + 'Predictions.csv')

    obs_pf = pd.DataFrame(test_y)
    obs_pf.to_csv(path_g + 'Observations.csv')

    test = all_scores(Traits,Traits,obs_pf, pred_df,samp_w_test)
    test.to_csv(path_g + 'Global_all.csv')

    if (GP):
        for t in range(len(Traits)):
            gap_fil = fill_gp(gap_fil, best_model, scaler_list, Traits, t)
        gap_fil.to_csv(path_g + 'Gapfil_allTraits.csv')

        #############Inexact ############   

        mx = pd.concat([db_train[Traits].quantile(0.99),db_test[Traits].quantile(0.99)],axis=1).T.max()
        s = (gap_fil.reset_index(drop=True)[Traits]>mx)
        idx = np.where(s)[0]

        mx = pd.concat([db_train[Traits].quantile(0.01),db_test[Traits].quantile(0.01)],axis=1).T.min()
        s = (gap_fil.reset_index(drop=True)[Traits]<mx)
        idx1 = np.where(s)[0]

        gap_fil = gap_fil.reset_index(drop=True).drop(list(idx) + list(idx1),axis=0)


        #### Final global model training

        path_g = dir_n + 'inexact/'
        create_path(path_g)

        ######### Train set ######

        train_x, train_y, samp_w_tr = data_prep('400', gap_fil, Traits, multi= True)
        samp_w_tr = samp_w_tr.loc[train_x.index]

            ##############
        scaler_list = save_scaler(train_y, save=True, dir_n = path_g, k = 'global')

        val_x = train_x.sample(frac = 0.1,random_state = seed)
        val_y = train_y.loc[val_x.index,:]
        samp_w_val = pd.DataFrame(samp_w_tr).sample(frac = 0.1,random_state = seed)

        if(samp_w_tr is not None):
            if (samp_w_tr.sum().sum() !=0):    
                train_ds = dataset(train_x.drop(val_x.index), train_y.drop(val_y.index), pd.DataFrame(samp_w_tr).drop(samp_w_val.index), scaler_list, Traits, shuffle=True,augment=True)
                test_ds = dataset(val_x, val_y, samp_w_val, scaler_list, Traits)
        else:
            train_ds = dataset(train_x.drop(val_x.index), train_y.drop(val_y.index), None, scaler_list, Traits, shuffle=True,augment=True)
            test_ds = dataset(val_x, val_y, None, scaler_list, Traits)

        ##### Model definition  and taraining #######
        input_shape = train_x.shape[1]
        output_shape = train_y.shape[1]

        EPOCHS = epochs 
        best_model = model_definition(input_shape, output_shape,lr = lr, kind= kind)

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,restore_best_weights=True)

        checkpoint_path = path_g + 'checkpoint'
        create_path(checkpoint_path)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_path + "/epoch{epoch:02d}-val_root_mean_squared_error{val_root_mean_squared_error:.2f}.hdf5",
        save_weights_only=True,
        monitor = 'val_root_mean_squared_error',
        mode='min',
        save_best_only=True)

        his = best_model.fit(train_ds,
                        validation_data = test_ds,
                        epochs = EPOCHS,
                        verbose=1, callbacks = [model_checkpoint_callback])

        val_acc_per_epoch = his.history['val_root_mean_squared_error']
        best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1    

        path_trial = path_g + "Model.json"
        path_best = checkpoint_path + "/epoch{0:02d}-val_root_mean_squared_error{1:.2f}.hdf5".format(best_epoch,min(val_acc_per_epoch))
        path_w = path_g + 'Trial_weights.h5'
        save_model(best_model, path_trial, path_best, path_w)


        test_x, test_y = data_prep('400', db_test, Traits, multi = True)

        pred = scaler_list.inverse_transform(best_model.predict(test_x))
        pred_df = pd.DataFrame(pred, columns = test_y.columns+ ' Predictions')
        pred_df.to_csv(path_g + 'Predictions.csv')

        obs_pf = pd.DataFrame(test_y)
        obs_pf.to_csv(path_g + 'Observations.csv')

        test = all_scores(Traits,Traits,obs_pf, pred_df,samp_w_test)
        test.to_csv(path_g + 'Global_all.csv')
