import pandas as pd
import numpy as np
import datetime

class Items:
    def __init__(self, data_path = '../data/'):
        self.items, self.category_tree = self.read_data(data_path)
        self.items.timestamp = self.format_date(self.items.timestamp)
        self.items = self.get_item_feature_interaction(self.items, self.category_tree)

    def read_data(self, data_path):
        return (
            pd.concat(
                [pd.read_csv(data_path+'item_properties_part1.csv')
                , pd.read_csv(data_path+'item_properties_part2.csv')]),
                pd.read_csv(data_path+'category_tree.csv')
                )
                
    def format_date(self, time_col):
        times =[]
        for i in time_col:
            times.append(datetime.datetime.fromtimestamp(i//1000.0)) 
        return times
    
    def get_item_feature_interaction(self, items, category_tree_df):
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