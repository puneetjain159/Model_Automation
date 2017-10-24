import os
import pandas as pd
import numpy as np
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

class Logger(object):
    def __init__(self):
        pass

    def info(self, msg):
        print 'INFO: {0}'.format(msg)


class VariableSelector():
    #Currently only support's H2o and LightGBM for generating Feature Importance
    #either input the pandas dataframe using dataset or specify CSV location
    #the Specify addtional categories  are the columns which need to be treated as Category
    def __init__(self, Algorithm = "LightGBM",dataset = "", Input_Dir = "", Target ="",
                 Exclude_columns = "", Num_features = 20, preprocess = True,log = None,
                 spcfy_cat = []):
        self.Algorithm = Algorithm
        self.dataset = dataset
        self.data = self.load_data(dataset,Input_Dir)
        self.Input_Dir = Input_Dir
        self.log = self.setup_log(log)
        self.log.info('Run initiated.')
        self.Target =Target
        self.Exclude_columns = Exclude_columns
        self.spcfy_cat = spcfy_cat
        self.cols_to_use = self.get_cols_to_use(self.data,Target,Exclude_columns)
        self.Cat_columns = self.get_Cat_columns(self.data,Target,spcfy_cat,self.cols_to_use)
        self.variable_importance  = self.get_variable_importance(Algorithm,self.data,Target,Input_Dir,
                                                                 Exclude_columns,preprocess,self.cols_to_use,
                                                                 spcfy_cat,self.Cat_columns)
        
    def get_cols_to_use(self,data,Target,Exclude_columns):
        if isinstance(Exclude_columns, str):
            Exclude_columns = list(Exclude_columns)  
        return list(set(data.columns)-set([Target])- set(Exclude_columns))
    
    def get_Cat_columns(self,data,Target,spcfy_cat,cols_to_use):
        return list(set(data[cols_to_use].select_dtypes(include=['object']).columns)
                                         .union(set(spcfy_cat)))
    
    
    def get_variable_importance(self,Algorithm,dataset,Target,Input_Dir,
                                Exclude_columns,preprocess,cols_to_use,
                                spcfy_cat,Cat_columns):
        data = self.load_data(dataset,Input_Dir)
        self.log.info('Dataset Loaded')   
        data = self.preprocess_data(preprocess,self.Algorithm,data,spcfy_cat,self.Cat_columns) 
        self.data = data
        return self.build_model(data,Algorithm,Target,cols_to_use,Cat_columns)

        
    def build_model(self,data,Algorithm,Target,cols_to_use,Cat_columns):
        if Algorithm in ["H2o"]:
            model = H2ORandomForestEstimator(model_id="Random_forest_FI",
                                            ntrees=400,
                                            stopping_rounds=2,
                                            score_each_iteration=True,
                                            seed=1000000)
            model.train(cols_to_use,y= Target, training_frame=data)
            self.log.info('Model Build')  
            return model._model_json['output']['variable_importances'].as_data_frame()
        else:
            train_data = lgb.Dataset( data[cols_to_use],data[Target] ,feature_name = cols_to_use,
                                     categorical_feature=Cat_columns)
            train_data.set_categorical_feature(Cat_columns)
            self.log.info(train_data) 
            params = {
                            'task': 'train',
                            'boosting_type': 'gbdt',
                            'objective': 'regression',
                            #'metric': {'l2', 'auc'},
                            #'num_leaves': 31,
                            'learning_rate': 0.1,
                            'feature_fraction': 0.9,
                            'bagging_fraction': 0.8,
                            'bagging_freq': 5,
                            'verbose': 0,
                           # 'categorical_feature' : ['name:' + str(col) for col in Cat_columns]
                        }
            if len(data[Target].unique()) == 2:
                params["objective"] = 'binary'
            elif len(data[Target].unique()) < 10 :
                params["objective"] = 'multiclass'
            model = lgb.train(params,
                            train_data,
                            num_boost_round=200,
                           # valid_sets=lgb_eval,
                           # early_stopping_rounds=25
                             )
            self.log.info(params) 
            self.log.info('Model Build')  
            importances = model.feature_importance()
            #print (importances.shape)
            imp_list = []
            for row in zip(data[cols_to_use].columns, map(lambda x:round(x,4), importances)):
                imp_list.append(row)
            return (pd.DataFrame(imp_list, columns=['Column', 'Importance'])).sort_values(['Importance'], ascending = False)

        
        
        
    def setup_log(self, log):
        if log is None:
            log = Logger()
        return log
    
    def load_data(self,data,Input_Dir):
        if isinstance(data, pd.DataFrame):
            return data
        else:
            return pd.read_csv(Input_Dir)
    
    
        
    def preprocess_data(self,preprocess,Algorithm,data,spcfy_cat,Cat_columns):
        if preprocess == False:
            self.log.info("No Preprocessing Done")
        elif  Algorithm in ["H2o"]:
            h2o.init()
            h2o.remove_all() 
            self.log.info('Preprocessed Data h2o frame') 
            data = h2o.H2OFrame(data)
            for col in spcfy_cat:
                data[col] = data[col].asfactor()

            return data
              
        else:
            for c in Cat_columns:
                lbl = LabelEncoder()
                lbl.fit(list(data[c].values) )
                data[c] = lbl.transform(list(data[c].values))
                self.log.info('Preprocessed Data pd frame')
            return data
        
            
if __name__ == '__main__':
    m = VariableSelector(Input_Dir = "/home/puneetj/CLA_Macro/MacroPy/data/german_data.csv",
                     Algorithm = "H2o",
                    # dataset = data,
                     Target ="tar",
                     preprocess = True,
                 #    Exclude_columns = ["status_sex"],
                  #  spcfy_cat = ["installment_rate_income"]
                    )
    print(m.variable_importance)
