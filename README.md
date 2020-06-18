# Machine learning tool

Flask api and the code to retrain the model, which requires data, both extracted out of SIA and some dumps out of old systems. For data, contact: m.sukel@amsterdam.nl

# installation
```
pip install -r requirements.txt
```


# input data

csv input file with at least the following columns:
| column  | description |
| ------------- | ------------- |
| Main  | Main category  |
| Middle  | Middle category  |
| Sub  | Sub category  |
| Text  | message  |


# training model
navigate to [app folder](https://github.com/Signalen/classification-endpoint/tree/master/app)
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
