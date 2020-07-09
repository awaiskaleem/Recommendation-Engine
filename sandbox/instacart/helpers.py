import numpy as np
import pandas as pd

def get_user_list(df, user_column):
    """
    
    creating a list of user from dataframe df, user_column is a column 
    consisting of users in the dataframe df
    
    """
    
    return np.sort(df[user_column].unique())

def get_item_list(df, item_name_column):
    
    """
    
    creating a list of items from dataframe df, item_column is a column 
    consisting of items in the dataframe df
    
    return to item_id_list and item_id2name_mapping
    
    """
    
    item_list = df[item_name_column].unique()
    
    
    return item_list

def get_feature_list(aisle_df, department_df, aisle_name_column, department_name_column):
    
    aisle = aisle_df[aisle_name_column]
    department = department_df[department_name_column]
    
    return pd.concat([aisle, department], ignore_index = True).unique()

# creating user_id, item_id, and features_id

def id_mappings(user_list, item_list, feature_list):
    """
    
    Create id mappings to convert user_id, item_id, and feature_id
    
    """
    user_to_index_mapping = {}
    index_to_user_mapping = {}
    for user_index, user_id in enumerate(user_list):
        user_to_index_mapping[user_id] = user_index
        index_to_user_mapping[user_index] = user_id
        
    item_to_index_mapping = {}
    index_to_item_mapping = {}
    for item_index, item_id in enumerate(item_list):
        item_to_index_mapping[item_id] = item_index
        index_to_item_mapping[item_index] = item_id
        
    feature_to_index_mapping = {}
    index_to_feature_mapping = {}
    for feature_index, feature_id in enumerate(feature_list):
        feature_to_index_mapping[feature_id] = feature_index
        index_to_feature_mapping[feature_index] = feature_id
        
        
    return user_to_index_mapping, index_to_user_mapping, \
           item_to_index_mapping, index_to_item_mapping, \
           feature_to_index_mapping, index_to_feature_mapping


def get_user_product_interaction(orders_df, order_products_train_df, order_products_test_df, products_df):
    
    # creating a dataframe consists of TWO columns user_id, and product_name (product bought by the user) for the train data
    user_to_product_train_df = orders_df[orders_df["eval_set"] == "prior"][["user_id", "order_id"]].\
    merge(order_products_train_df[["order_id", "product_id"]]).merge(products_df[["product_id", "product_name"]])\
    [["user_id", "product_name"]].copy()
    
    # giving rating as the number of product purchase count
    user_to_product_train_df["product_count"] = 1
    user_to_product_rating_train = user_to_product_train_df.groupby(["user_id", "product_name"], as_index = False)["product_count"].sum()
    
    # creating a dataframe consists of TWO columns user_id, and product_name (product bought by the user) for the test data
    user_to_product_test_df = orders_df[orders_df["eval_set"] == "train"][["user_id", "order_id"]].\
    merge(order_products_test_df[["order_id", "product_id"]]).merge(products_df[["product_id", "product_name"]])\
    [["user_id", "product_name"]].copy()
    
    # giving rating as the number of product purchase count (including the previous purchase in the training data)
    user_to_product_test_df["product_count"] = 1
    user_to_product_rating_test = user_to_product_test_df.groupby(["user_id", "product_name"], as_index = False)["product_count"].sum()
    
    # merging with the previous training user_to_product_rating_training
    
    user_to_product_rating_test = user_to_product_rating_test.\
    merge(user_to_product_rating_train.rename(columns = {"product_count" : "previous_product_count"}), how = "left").fillna(0)
    user_to_product_rating_test["product_count"] = user_to_product_rating_test.apply(lambda x: x["previous_product_count"] + \
                                                                                    x["product_count"], axis = 1)
    user_to_product_rating_test.drop(columns = ["previous_product_count"], inplace = True)
    
    return user_to_product_rating_train, user_to_product_rating_test

def get_interaction_matrix(df, df_column_as_row, df_column_as_col, df_column_as_value, row_indexing_map, 
                          col_indexing_map):
    
    row = df[df_column_as_row].apply(lambda x: row_indexing_map[x]).values
    col = df[df_column_as_col].apply(lambda x: col_indexing_map[x]).values
    value = df[df_column_as_value].values
    
    return coo_matrix((value, (row, col)), shape = (len(row_indexing_map), len(col_indexing_map)))

def get_product_feature_interaction(product_df, aisle_df, department_df, aisle_weight = 1, department_weight = 1):
    item_feature_df = product_df.merge(aisle_df).merge(department_df)[["product_name", "aisle", "department"]]
    
    # start indexing
    item_feature_df["product_name"] = item_feature_df["product_name"]
    item_feature_df["aisle"] = item_feature_df["aisle"]
    item_feature_df["department"] = item_feature_df["department"]
    
    # allocate aisle and department into one column as "feature"
    
    product_aisle_df = item_feature_df[["product_name", "aisle"]].rename(columns = {"aisle" : "feature"})
    product_aisle_df["feature_count"] = aisle_weight # adding weight to aisle feature
    product_department_df = item_feature_df[["product_name", "department"]].rename(columns = {"department" : "feature"})
    product_department_df["feature_count"] = department_weight # adding weight to department feature
    
    # combining aisle and department into one
    product_feature_df = pd.concat([product_aisle_df, product_department_df], ignore_index=True)
    
    # saving some memory
    del item_feature_df
    del product_aisle_df
    del product_department_df
    
    
    # grouping for summing over feature_count
    product_feature_df = product_feature_df.groupby(["product_name", "feature"], as_index = False)["feature_count"].sum()
    
    
    return product_feature_df