import numpy as np 
import pandas as pd
import datetime
from sklearn import preprocessing

class Interactions:
    def __init__(self
    ,data_path="./data/"
    ,start_date='2015-05-03'
    ,end_date='2015-08-30'
    , split_ratio = 0.75
    , user_col = 'visitorid'
    , item_col = 'itemid'): 
        '''
        Purpose: This class processes events data in e-commerce dataset
        '''
        self.data_path = data_path
        self.events = pd.DataFrame()
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.start_date = start_date
        self.end_date = end_date
        self.user_col = user_col
        self.item_col = item_col
        self.split_ratio = split_ratio
        self.popular_items = []
        
    def fetch_events(self):
        '''
        Purpose: read events, format timestamp column from unix format to datetime format
        '''
        self.events = pd.read_csv(self.data_path+'events.csv')
        self.events = self.events.drop_duplicates(subset=['timestamp', 'visitorid','itemid'], keep = 'first')
        
        # self.events['sequence'] = self.events['event'].apply(lambda x: 1 if x=='view' else 2 if x=='addtocart' else 3 if x=='transaction' else null)
        # self.events = self.events.sort_values('sequence').drop_duplicates(['visitorid','itemid'], keep='last')

        self.events = self.events.assign(date=pd.Series(datetime.datetime.fromtimestamp(i/1000).date() for i in self.events.timestamp))
        self.events = self.events.sort_values('date').reset_index(drop=True)
        fd = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date()
        self.events = self.events[(self.events.date >= fd(self.start_date)) & (self.events.date <= fd(self.end_date))]
        self.events = self.events[['visitorid','itemid','event', 'date']]

    def train_test_split(self):
        '''
        Purpose: Split events into train and test split given split_ratio
        '''
        split_point = np.int(np.round(self.events.shape[0]*self.split_ratio))
        self.train = self.events.iloc[0:split_point]
        self.test = self.events.iloc[split_point::]

    def processing_testset(self):
        '''
        Purpose: Keeps only those users and items in test set that have some history in train
        '''
        self.train =  self.train[
            (self.train['visitorid'].isin(self.test['visitorid'])) & 
            (self.train['itemid'].isin(self.test['itemid']))
        ]

        self.test =  self.test[
            (self.test['visitorid'].isin(self.train['visitorid'])) & 
            (self.test['itemid'].isin(self.train['itemid']))
        ]

        
    def compute_ratings(self):
        cat_rating =preprocessing.LabelEncoder()
        self.train['rating'] = cat_rating.fit_transform(self.train.event)
        self.test ['rating'] = cat_rating.transform(self.test.event)
        self.events ['rating'] = cat_rating.fit_transform(self.events.event)

    def run_unit_tests(self):
        '''
        Purpose: Unit Test to ensure there are no user or items that are in test but not in train
        '''
        try:
            assert(
            len(self.test[(self.test['visitorid'].isin(self.train['visitorid'])==False)])==0
            & len(self.test[(self.test['itemid'].isin(self.train['itemid'])==False)])==0
              )
        except:
            raise Exception( "Train/Test split failed, make sure users and items in test are already present in train" )

    def get_popular_items(self):
        self.popular_items = list(self.train.groupby([self.item_col], as_index = False)["rating"].sum().sort_values(by='rating', ascending = False)[self.item_col])

class Items:
    def __init__(self, item_col = 'itemid', feat_col = 'feature'):
        '''
        Purpose: Extract Items and their features
        '''
        self.data_path = './data/'
        self.items = pd.DataFrame()
        self.category_tree = pd.DataFrame()
        self.item_interactions = pd.DataFrame()
        self.item_col = item_col
        self.feat_col = feat_col
   
    def fetch_items(self):
        self.items = pd.concat(
            [pd.read_csv(self.data_path+'item_properties_part1.csv')
            , pd.read_csv(self.data_path+'item_properties_part2.csv')])
        self.items = self.items.drop_duplicates(subset=['itemid','property','value'], keep = 'first')
        times =[]
        for i in self.items.timestamp:
            times.append(datetime.datetime.fromtimestamp(i//1000.0))
        self.items.timestamp = times
        self.category_tree = pd.read_csv(self.data_path+'category_tree.csv')
        self.category_tree.loc[self.category_tree.parentid!=self.category_tree.parentid, 'parentid'] = 0
        
    
    def get_item_feature_interaction(self):
        '''
        Purpose: Extract Item Features
        '''
        #category property
        items_to_cat = self.items[(self.items.property == 'categoryid')][['itemid','value']].drop_duplicates()
        items_to_cat['value'] = items_to_cat['value'].astype(int)
        item_feature_df = pd.merge(items_to_cat, self.category_tree.rename(columns={'categoryid':'value'}), on='value',  how='left')

        item_category_df = item_feature_df[["itemid", "value"]].rename(columns = {"value" : "feature"})
        item_category_df["feature_count"] = 1 # adding weight to category feature
        item_parent_df = item_feature_df[["itemid", "parentid"]].rename(columns = {"parentid" : "feature"})
        item_parent_df["feature_count"] = 1 # adding weight to department feature

        item_feature_df_sub = pd.concat([item_category_df, item_parent_df], ignore_index=True)

        # saving some memory
        del item_feature_df
        del item_category_df
        del item_parent_df

        # grouping for summing over feature_count
        self.items = item_feature_df_sub.groupby(["itemid", "feature"], as_index = False)["feature_count"].sum()
    
    def run_unit_tests(self):
        '''
        Purpose: Unit Test to ensure there are no user or items that are in items but not in interactions
        '''
        try:
            assert(
            len(self.items[(self.items['itemid'].isin(self.items['itemid'])==False)])==0
              )
        except:
            raise Exception( "Items found that are not in interactions. Clean up items first" )
