import joblib
import numpy as np

if __name__ == '__main__':
    print('loading models...')
    sub_cat = joblib.load('sub_model.pkl')
    sub_categories = joblib.load('sub_slugs.pkl')

    while(True):
        text = input("input (or ctrl+c to stop):") 

        a = sub_cat.predict_proba([text])
        z = int(np.argmax(a, axis=1))
        r = sub_categories[z],max(a[0])
        print(r)
