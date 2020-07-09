if 'etl.preprocessing' in sys.modules:  
    print("Replaced")
    del sys.modules["etl.preprocessing"]

from etl.feature_extractor import Items, Interactions
from etl.preprocessing import Preprocessor

class Model:
    def __init__(self):
        self.items = Items()
        self.interactions = Interactions()
        self.model_data = Preprocessor(self.items, self.interactions, 'visitorid','itemid')
        self.model = LightFM(no_components=20, loss='warp', learning_rate=0.01)
        self.model_without_items = self.fit_model(self.model, self.model_data)
        self.display_results(self.model_without_items, self.model_data)
        self.model_with_items = self.fit_model(self.model, self.model_data, item_flg = 'items')
        self.display_results(self.model_without_items, self.model_data)
        self.model_with_items

    def fit_model(self, model, model_data, item_flg = ''):
        if item_flg == 'items':
            print("Training model WITH items")
            return model.fit(
                model_data.rate_matrix['train']
                , item_features = model_data.rate_matrix['feature']
                , epochs=50, num_threads=8)
        else:
            print("Training model WITHOUT items")
            return model.fit(
                model_data.rate_matrix['train']
                , item_features = None
                , epochs=50, num_threads=8)

    def display_results(self, model, model_data):
        auc_without_features = auc_score(model = model,
        test_interactions = model_data.rate_matrix['test'],
        num_threads = 4, check_intersections = False)
        print("average AUC without adding item-feature interaction = {0:.{1}f}".format(auc_without_features.mean(), 2))
