import joblib
import numpy as np
import os
import logging
import warnings

from settings import SIGNALS_CATEGORY_URL, MODELS_DIRECTORY

warnings.filterwarnings("ignore",category=DeprecationWarning)
class MoraCategoryClassifier:
    
    """
    
    Classification models

    string to categories and probabilities
    
    """

    def __init__(self):
        self.createLogger()
        self.logger.info("MoraCategoryClassifier: Init")
        self.has_valid_models = False
        try:
            self.main_cat = joblib.load(os.path.join(MODELS_DIRECTORY, 'main_model.pkl'))
            self.categories = joblib.load(os.path.join(MODELS_DIRECTORY, 'main_slugs.pkl'))

            self.sub_cat = joblib.load(os.path.join(MODELS_DIRECTORY, 'sub_model.pkl'))
            self.sub_categories = joblib.load(os.path.join(MODELS_DIRECTORY, 'sub_slugs.pkl'))
            self.has_valid_models = True
        except FileNotFoundError as ex:
            self.logger.warn("Faild to init ML model: {}".format(str(ex)))

    # ----
    
    def createLogger(self):
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.INFO) # SETTING: log level
        
        # logger handlers
        handler = logging.StreamHandler()
        # handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)-4s %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    # ----

    def classifyCategoryWithProbability(self,zin):
        '''
        input string, returns best category, and probability
        '''
        
        
        a = self.main_cat.predict_proba([zin])
        z = int(np.argmax(a, axis=1))
        return self.categories[z],max(a[0])
    
    # ----

    def classifyAllCategoriesWithProbability(self,zin):
        '''
        input string, returns all category and probability
        '''
        if not self.has_valid_models:
            return None

        a = self.main_cat.predict_proba([zin])
        probs = list(reversed(sorted(a[0])))
        
        cats = ["{prefix}{cat}".format(prefix=SIGNALS_CATEGORY_URL, cat=self.categories[z]) for z in list(reversed(np.argsort(a)[::-1][0][-100:]))]
        
        return cats, probs

    # ----
    
    def classifySubCategoryWithProbability(self,zin):
        '''
        input string, returns best subcategory, and probability
        '''

        a = self.sub_cat.predict_proba([zin])
        z = int(np.argmax(a, axis=1))
        return self.sub_categories[z],max(a[0])
    
    # ----

    def classifyAllSubCategoriesWithProbability(self,zin):
        '''
        input string, returns all subcategory and probability
        '''
        if not self.has_valid_models:
            return None

        a = self.sub_cat.predict_proba([zin])
        probs = list(reversed(sorted(a[0])))
        cats = ["{prefix}{cat}".format(prefix=SIGNALS_CATEGORY_URL, cat=self.sub_categories[z]) for z in list(reversed(np.argsort(a)[::-1][0][-100:]))]
      

        return cats, probs

    # ----

