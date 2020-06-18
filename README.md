# Machine learning tool

Flask api and the code to retrain the model, which requires data, both extracted out of SIA and some dumps out of old systems. For data, contact: m.sukel@amsterdam.nl

To retrain, use the [classification/ML](https://github.com/Signalen/classification/) cmd line tool.

# installation
```
pip install -r requirements.txt
```

# training model
See python train.py for all options
```
python train.py --csv file.csv
```
Rename resulting files to "main_model.pkl, sub_model.pkl, main_slugs.pkl, sub_slugs.pkl"

# running service

To load new model into flask (copy into app folder)
| file  | description |
| ------------- | ------------- |
| main_model.pkl  | model for main category |
| sub_model.pkl  | model for sub category |
| main_slugs.pkl | slugs for main category |
| sub_slugs.pkl | slugs for sub category  |

```
run docker-compose build
```

To activate the flask api run:
```
docker-compose up -d
```

To test the current loaded model, open web_pages/index.html or POST "text" to http://localhost:8140/signals_mltool/predict with the flask app running.
