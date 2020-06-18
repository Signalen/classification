# Machine learning tool

Flask api and the code to retrain the model, which requires data, both extracted out of SIA and some dumps out of old systems. For data, contact: m.sukel@amsterdam.nl

To retrain, run python create_pickle/retrain.py. This will automatically create new pickles for all classes that are active in SIA, are not 'overig' and have n>50.

To load new model into flask, copy main_model.pkl, sub_model.pkl, sub_slugs.pkl and main_slugs.pkl to flask_demo and run docker-compose build.

To activate the flask api run docker-compose up -d.

To test the current loaded model, open web_pages/index.html or POST "text" to http://localhost:8140/signals_mltool/predict with the flask app running.
