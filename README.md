# Machine learning tool

Flask api and the code to retrain the model, which requires data, both extracted out of SIA and some dumps out of old systems.

# installation (ML train tool)
**Deprecated: use docker instead**

```
pip install -r requirements-train.txt
```

# installation
**Deprecated: use docker instead**

Use the requirements.txt to run (flask) endpoint locally. This step can be skipped if you are using the docker container.
```
pip install -r requirements.txt
```

# Building the Docker images
Navigate to the root directory and pull the  relevant images and build the services:

```console
docker-compose pull
docker-compose build
```

# input data

The `CSV` input file must have at least the following columns:

| column      | description   |
|-------------|---------------|
| Text        | message       |
| Main        | Main category |
| Sub         | Sub category  |

The columns must be in the order `Text,Main,Sub`, no header is required.


# Training model using docker compose

To train the model run the following command:

```
docker-compose run --rm train python train.py --csv=/input/{name of csv file} --columns={name of column}
```

for example:

```
docker-compose run --rm train python train.py --csv=/input/dump.csv --columns=Main
```

The files will be saved in the `ouput` directory.

In the example above this would result in:
- `/output/Main_model.pkl`
- `/output/Main_slugs.pkl`
- `/output/Main_dl.csv`
- `/output/Main_matrix.csv`
- `/output/Main_matrix.pdf`

The `pkl` files can be used for the classification endpoint.  
And the confusion matrix will be available as `pdf` and `csv`.


# training model

navigate to [app folder](https://github.com/Signalen/classification-endpoint/tree/master/app)
See python train.py for all options. 

To train Middle and Sub categoeries use:
```
python train.py --csv file.csv --columns Middle,Sub
```
This step will generate a categories `json` file. Use this file to load the categories in the backend.
```
python manage.py load_categories <file.json>
```

To train Middle category use:
```
python train.py --csv file.csv --columns Middle
```

Rename resulting files to "main_model.pkl, sub_model.pkl, main_slugs.pkl, sub_slugs.pkl" and copy the pkl files into the classification endpoint.

# running service

To load new model into flask (copy into app folder)

| file           | description             |
|----------------|-------------------------|
| main_model.pkl | model for main category |
| sub_model.pkl  | model for sub category  |
| main_slugs.pkl | slugs for main category |
| sub_slugs.pkl  | slugs for sub category  |

```
run docker-compose build
```

To activate the flask api run:
```
docker-compose up -d
```

To test the current loaded model, open web_pages/index.html or POST "text" to http://localhost:8140/signals_mltool/predict with the flask app running.
