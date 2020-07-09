import numpy as np 
import pandas as pd
import datetime

class Interactions:
    def __init__(self,path_to_events="../data/events.csv",start_date='2015-5-3',end_date='2015-5-18', split_ratio = 0.8): 
        '''
        Purpose: This class processes events data in e-commerce dataset
        '''
        self.events = self.read_events(path_to_events)
        self.events = self.filter_dates(self.events,start_date,end_date)
        self.events = self.process_rating(self.events)
        self.train, self.test = self.train_test_split(self.events, split_ratio)
        del self.events
        # self.test = self.processing_testset(self.test, self.train)
        # self.run_unit_tests(self.test, self.train)
        
        #Get only visitors and items that are in train as well
        #This is for testing so we have true positives

    def process_rating(self, events):
        events['rating'] = events['event'].apply(lambda x: 1 if x=='view' else 2 if x=='addtocart' else 3 if x=='transaction' else null)
        return events.sort_values('rating').drop_duplicates(['visitorid','itemid'], keep='last')[['visitorid','itemid','rating']]
        
    def read_events(self, path_to_events):
        '''
        Purpose: read events, format timestamp column from unix format to datetime format
        '''
        events = pd.read_csv(path_to_events)
        events = events.assign(date=pd.Series(datetime.datetime.fromtimestamp(i/1000).date() for i in events.timestamp))
        events = events.sort_values('date').reset_index(drop=True)
        return events[['visitorid','itemid','event', 'date']]
    
    def filter_dates(self, events, start_date, end_date):
        '''
        Purpose: Filters out date except given dates
        '''
        fd = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date()
        return events[(events.date >= fd(start_date)) & (events.date <= fd(end_date))]
    
    def train_test_split(self, events, split_ratio):
        '''
        Purpose: Split events into train and test split given split_ratio
        '''
        split_point = np.int(np.round(events.shape[0]*split_ratio))
        return events.iloc[0:split_point], events.iloc[split_point::]
    
    def processing_testset(self, test, train):
        '''
        Purpose: Keeps only those users and items in test set that have some history in train
        '''
        return test[
            (test['visitorid'].isin(train['visitorid'])) & 
            (test['itemid'].isin(train['itemid']))
        ]
    
    def run_unit_tests(self, test, train):
        '''
        Purpose: Unit Test to ensure there are no user or items that are in test but not in train
        '''
        try:
            assert(
            len(test[(test['visitorid'].isin(train['visitorid'])==False)])==0
            & len(test[(test['itemid'].isin(train['itemid'])==False)])==0
              )
        except:
            raise Exception( "Train/Test split failed, make sure users and items in test are already present in train" )


class Items:
    def __init__(self, data_path = '../data/'):
        '''
        Purpose: Extract Items and their features
        '''
        self.items, self.category_tree = self.read_data(data_path)
        self.items.timestamp = self.format_date(self.items.timestamp)
        self.items = self.get_item_feature_interaction(self.items, self.category_tree)

    def read_data(self, data_path):
        '''
        Purpose: Read csvs
        '''
        return (
            pd.concat(
                [pd.read_csv(data_path+'item_properties_part1.csv')
                , pd.read_csv(data_path+'item_properties_part2.csv')]),
                pd.read_csv(data_path+'category_tree.csv')
                )
                
    def format_date(self, time_col):
        '''
        Purpose: format date columns from unix format to datetime
        '''
        times =[]
        for i in time_col:
            times.append(datetime.datetime.fromtimestamp(i//1000.0)) 
        return times
    
    def get_item_feature_interaction(self, items, category_tree_df):
        '''
        Purpose: Extract Item Features
        '''
        #category property
        items_to_cat = items[(items.property == 'categoryid')][['itemid','value']].drop_duplicates()
        items_to_cat['value'] = items_to_cat['value'].astype(int)
        item_feature_df = pd.merge(items_to_cat, category_tree_df.rename(columns={'categoryid':'value'}), on='value',  how='left')

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
        return item_feature_df_sub.groupby(["itemid", "feature"], as_index = False)["feature_count"].sum()
    
    def cleanup_items(self, items, intrctions):
        '''
        Purpose: Keeps only those users and items in items that have some history in interactions
        '''
        return items[
            (items['itemid'].isin(intrctions['itemid']))
        ]
    def run_unit_tests(self, items, intrctions):
        '''
        Purpose: Unit Test to ensure there are no user or items that are in items but not in interactions
        '''
        try:
            assert(
            len(items[(items['itemid'].isin(intrctions['itemid'])==False)])==0
              )
        except:
            raise Exception( "Items found that are not in interactions. Clean up items first" )
