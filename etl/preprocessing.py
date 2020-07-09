import numpy as np
from scipy.sparse import coo_matrix
from sklearn import preprocessing

class Preprocessor:
    def __init__(self, items, interactions, user_col = 'visitorid', item_col = 'itemid'):
        self.items = items
        self.interactions = interactions
        self.user_col = user_col
        self.item_col = item_col

        self.items, self.interactions = self.apply_business_logic(self.items, self.interactions)
        
        self.items.run_unit_tests(self.items.items, self.interactions.train)
        self.interactions.run_unit_tests(self.interactions.test, self.interactions.train)
        
        self.user_list = np.unique(self.interactions.train[user_col])
        self.item_list = np.unique(self.items.items[item_col])
        self.feat_list = np.unique(self.items.items['feature'])
        
        
        self.trans_cat_train,self.trans_cat_test, self. items_cat, self.cate_enc_dict = self.premodel_processing(self.items, self.interactions, self.user_list, self.item_list)
        self.rate_matrix = self.create_matrices(self.items, self.interactions, self.trans_cat_train, self.trans_cat_test, self.items_cat, self.user_list, self.item_list, self.feat_list)

        
    def apply_business_logic(self, items, interactions):
        items.items = items.cleanup_items(items.items, interactions.train)
        interactions.train = items.cleanup_items(interactions.train, items.items)
        interactions.test = items.cleanup_items(interactions.test, items.items)
        interactions.test = interactions.processing_testset(interactions.test, interactions.train)
        return items, interactions

    def premodel_processing(self, items, interactions, user_list, item_list):
        id_cols=[self.user_col,self.item_col]
        trans_cat_train=dict()
        trans_cat_test=dict()
        items_cat = dict()
        cate_enc_dict = dict()

        cate_enc_dict[self.user_col] = preprocessing.LabelEncoder().fit(user_list)
        cate_enc_dict[self.item_col] = preprocessing.LabelEncoder().fit(item_list)
        cate_enc_dict['feature'] = preprocessing.LabelEncoder().fit(items.items.feature)

        for k in id_cols:
            trans_cat_train[k]=cate_enc_dict[k].transform(interactions.train[k].values)
            trans_cat_test[k]=cate_enc_dict[k].transform(interactions.test[k].values)
        
        items_cat['itemid'] = cate_enc_dict['itemid'].transform(items.items['itemid'].values)
        items_cat['feature'] = cate_enc_dict['feature'].transform(items.items['feature'].values)

        return trans_cat_train,trans_cat_test, items_cat, cate_enc_dict

    def create_matrices(self, items, interactions, trans_cat_train, trans_cat_test, items_cat, user_list, item_list, feat_list):
        rate_matrix = dict()
        rate_matrix['train'] = coo_matrix((interactions.train['rating'], (trans_cat_train['visitorid'], trans_cat_train['itemid'])), shape=(len(user_list),len(item_list)))
        
        rate_matrix['test'] = coo_matrix(
            (interactions.test['rating']
            , (trans_cat_test['visitorid'], trans_cat_test['itemid']))
            , shape=(len(user_list),len(item_list)))
        
        rate_matrix['feature'] = coo_matrix(
            (items.items['feature_count']
            , (items_cat['itemid'], items_cat['feature']))
            , shape=(len(item_list),len(feat_list)))
        
        return rate_matrix