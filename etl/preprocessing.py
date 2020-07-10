import numpy as np
from scipy.sparse import coo_matrix
from sklearn import preprocessing

class Preprocessor:
    def __init__(self):
        self.user_list = []
        self.item_list = []
        self.feat_list = []

        self.trans_cat_all=dict()
        self.trans_cat_train=dict()
        self.trans_cat_test=dict()
        self.items_cat = dict()
        self.cate_enc_dict = dict()
        self.rate_matrix = dict()

    def get_lists(self,items, interactions):
        self.user_list = np.unique(interactions.events[interactions.user_col])
        self.item_list = np.unique(interactions.events[items.item_col])
        self.feat_list = np.unique(items.items[items.feat_col])
        

    def apply_business_logic(self, items, interactions):
        #All items in train should be available in items
        # interactions.events = interactions.events[(interactions.events['itemid'].isin(items.items['itemid']))]
        # interactions.train = interactions.train[(interactions.train['itemid'].isin(items.items['itemid']))]
        
        #All in test should be available in train
        interactions.test =  interactions.test[
            (interactions.test['visitorid'].isin(interactions.train['visitorid'])) & 
            (interactions.test['itemid'].isin(interactions.train['itemid']))
        ]
        # DO NOT NEED: All in items should be there in train
        items.items = items.items[(items.items['itemid'].isin(interactions.events['itemid']))]
        # DO NOT NEED: All in test should be there in items
        # interactions.test = interactions.test[(interactions.test['itemid'].isin(items.items['itemid']))]
        return items, interactions

    def premodel_processing(self, items, interactions, feature_col = 'features'):
        id_cols=[interactions.user_col,items.item_col]
        for k in id_cols:
            self.cate_enc_dict[k] = preprocessing.LabelEncoder().fit(interactions.events[k].values)
            self.trans_cat_all[k]=self.cate_enc_dict[k].transform(interactions.events[k].values)
            self.trans_cat_train[k]=self.cate_enc_dict[k].transform(interactions.train[k].values)
            self.trans_cat_test[k]=self.cate_enc_dict[k].transform(interactions.test[k].values)
        
        self.cate_enc_dict['feature'] = preprocessing.LabelEncoder().fit(self.feat_list)
        self.items_cat[items.item_col] = self.cate_enc_dict[items.item_col].transform(items.items[items.item_col])
        self.items_cat['feature'] = self.cate_enc_dict['feature'].transform(items.items['feature'])

    def create_matrices(self, items, interactions):
        
        self.rate_matrix['all'] = coo_matrix(
            (interactions.events['rating']
            , (self.trans_cat_all['visitorid'], self.trans_cat_all['itemid']))
            , shape=(len(self.user_list),len(self.item_list)))
        
        self.rate_matrix['train'] = coo_matrix(
            (interactions.train['rating']
            , (self.trans_cat_train['visitorid'], self.trans_cat_train['itemid']))
            , shape=(len(self.user_list),len(self.item_list)))
        
        self.rate_matrix['test'] = coo_matrix(
            (interactions.test['rating']
            , (self.trans_cat_test['visitorid'], self.trans_cat_test['itemid']))
            , shape=(len(self.user_list),len(self.item_list)))
        
        self.rate_matrix['feature'] = coo_matrix(
            (items.items['feature_count']
            , (self.items_cat['itemid'], self.items_cat['feature']))
            , shape=(len(self.item_list),len(self.feat_list)))
        