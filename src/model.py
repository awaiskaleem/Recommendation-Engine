import sys
import csv
from lightfm import LightFM
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
from etl.feature_extractor import Items, Interactions
from etl.preprocessing import Preprocessor
import pickle
from scipy import sparse
import joblib
import pandas as pd
import numpy as np

class Model:
    def __init__(self, model_path = './artifacts/', data_path = './data/'):
        self.items = Items()
        self.interactions = Interactions()
        self.model_data = Preprocessor()
        self.model_with_items = LightFM()
        self.model_without_items = LightFM()
        self.model_path = model_path
        self.data_path = data_path
        
    def load_training_data(self):
        #Items
        print("Loading Items")
        self.items.fetch_items()
        self.items.get_item_feature_interaction()
        self.items.run_unit_tests()
        #Interactions
        print("Loading Interactions")
        self.interactions = Interactions()
        self.interactions.fetch_events()
        self.interactions.train_test_split()
        self.interactions.processing_testset()
        self.interactions.run_unit_tests()
        self.interactions.compute_ratings()
        self.interactions.get_popular_items()
        
        #Model Data
        print("Preparing Model Data")
        self.model_data = Preprocessor()
        self.items, self.interactions = self.model_data.apply_business_logic(self.items, self.interactions)
        self.model_data.get_lists(self.items, self.interactions)
        self.model_data.premodel_processing(self.items, self.interactions)
        self.model_data.create_matrices(self.items, self.interactions)

    def train(self):
        print("Training")
        self.model_without_items = LightFM(no_components=30, loss='warp', learning_rate = 0.01)
        self.model_without_items = self.fit_model(self.model_without_items, self.model_data)
        self.display_results(self.model_without_items, self.model_data)
        
        self.model_with_items = LightFM(no_components=30, loss='warp', learning_rate = 0.01)
        self.model_with_items = self.fit_model(self.model_with_items, self.model_data, item_flg = 'items')
        self.display_results(self.model_with_items, self.model_data , item_flg = 'items')
        
        print("Training with all")
        self.model_without_items = self.fit_model(self.model_without_items, self.model_data, stage = "all")
        self.model_with_items = self.fit_model(self.model_with_items, self.model_data, item_flg = 'items', stage = "all")

        print("Saving Models")
        self.save_models(self.model_without_items,'model_without_items')
        self.save_models(self.model_with_items,'model_with_items')
        self.save_models(self.model_data.cate_enc_dict['visitorid'],'cate_enc_dict_visitorid')
        self.save_models(self.model_data.cate_enc_dict['itemid'],'cate_enc_dict_itemid')
        self.save_models(self.model_data.cate_enc_dict['feature'],'cate_enc_dict_feature')
        print("Saving Matices")
        self.save_matrices(self.model_data.rate_matrix)

    def fit_model(self, model, model_data, item_flg = '', stage = "train"):
        item_features= None
        result_string = "Training model WITHOUT items"
        if item_flg == 'items':
            result_string = "Training model WITH items"
            item_features = model_data.rate_matrix['feature']

        return model.fit(
            model_data.rate_matrix[stage]
            , item_features = item_features
            , epochs=100, num_threads=8)
            
    
    def display_results(self, model, model_data, item_flg = 'without'):
        result_string = "without"
        item_feats = None
        if item_flg=='items':
            item_feats = model_data.rate_matrix['feature']
            result_string = 'with'
        
        auc = auc_score(model = model,
        test_interactions = model_data.rate_matrix['test'],
        item_features = item_feats,
        num_threads = 8)

        top_k = 100

        recall = recall_at_k(model = model, 
                                test_interactions = model_data.rate_matrix['test'],
                                item_features = item_feats,
                                num_threads = 8,
                                k = top_k)

        print("average AUC "+ result_string +" adding item-feature interaction = {0:.{1}f}".format(auc.mean(), 2))
        print("average Recall at "+ str(top_k) +" is "+ result_string +" adding item-feature interaction = {0:.{1}f}".format(recall.mean(), 2))

    def save_models(self, model, filename):
        with open(self.model_path+filename+'.pickle', 'wb') as fle:
            pickle.dump(model, fle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_matrices(self, matrix):
        for matrix_name in matrix:
            print("Saving Matrix"+matrix_name)
            sparse.save_npz(self.model_path+'rate_matrix_'+matrix_name+'.npz', matrix[matrix_name])

    def load_models(self, filename):
        print("Pickel Load Model: "+filename)
        with open(self.model_path+filename+'.pickle', 'rb') as fid:
            return pickle.load(fid)

    def load_matrices(self,matrix_name):
        print("Loading Matrix: "+matrix_name)
        self.model_data.rate_matrix[matrix_name] = sparse.load_npz(self.model_path+'rate_matrix_'+matrix_name+'.npz')

    def predict_recom(self, user_id_orig, recom_num, model, verbose = True):
        user_id = self.model_data.cate_enc_dict['visitorid'].transform([user_id_orig])

        #Known Positives
        known_positives = self.interactions.events['itemid'].values[np.where(self.model_data.trans_cat_all['visitorid']==user_id)]
        
        #Recommended
        scores = model.predict(user_ids = [user_id], item_ids = self.model_data.rate_matrix['feature'].col)
        top_items = self.interactions.events['itemid'].values[np.argsort(-scores)]
        if (verbose == True):
            self.print_recommendations(user_id_orig, known_positives, top_items, recom_num)# printing out the result
        
        return list(top_items[:recom_num])

    def print_recommendations(self, user_id, known_positives, top_items, recom_num):
        print("User %s" % user_id)
        print("     Known positives:")
        for x in known_positives[:recom_num]:
            print("                  %s" % x)

        print("     Recommended:")
        for x in top_items[:recom_num]:
            print("                  %s" % x)

    def get_predictions(self, recom_num, usr, model):
        print("Predicting for user:",usr)
        result_list = []
        if (self.interactions.events[self.interactions.events['visitorid']==usr].shape[0]==0):
            print("User Not Found, printing top ",recom_num," Recoms")
            result_list = self.interactions.popular_items[:recom_num]
        else:
            print("Existing User. Personalizing...")
            result_list = self.predict_recom(usr, recom_num, model)
        return result_list

    def predict_file(self, recom_num, model, pred_filename = 'predictions'):
        pred_df = pd.read_csv(self.data_path+pred_filename+'.csv')
        header = list(pred_df.columns)
        all_users = list(pred_df['visitorid'])

        with open('./output/results.csv', 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
                wr.writerow(header)

        with open('./output/results.csv', 'a+', newline='') as myfile:
            for usr in all_users:
                if (self.interactions.train[self.interactions.train[self.interactions.user_col]==usr].shape[0]==0):
                    result_list = self.interactions.popular_items[:recom_num]
                else:
                    result_list = self.predict_recom(usr, recom_num, model,verbose = False)
                result_list.insert(0, usr)
                wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
                wr.writerow(result_list)


        # for usr in np.array(pred_df[self.interactions.user_col])[:200]:
        #     result_list = []
        #     if (self.interactions.train[self.interactions.train[self.interactions.user_col]==usr].shape[0]==0):
        #         result_list = self.interactions.popular_items[:recom_num]
        #     else:
        #         result_list = self.predict_recom(usr, recom_num, model,verbose = False)
        #     for i in np.arange(recom_num):
        #         pred_df.loc[pred_df['visitorid'] == usr, ["item_"+str(i)]] = round(result_list[i])
        # pred_df.head(200).to_csv('./output/results.csv', index=False, float_format='%.f')
