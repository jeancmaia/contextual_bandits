poetry_install:
	poetry install 

raise_server:
	poetry run jupyter lab --port=8118

markdown_jupyter:
	poetry run jupyter nbconvert notebooks/ContextualBandits.ipynb --to markdown --output-dir='reports/' 

black:
	poetry run black .

generate_dataset:
	poetry run python scripts/data/make_dataset.py

install_agent:
	poetry run python setup.py install
	poetry run python setup.py clean --all install

train_model:
	poetry run python scripts/models/train_model.py

raise_fastapi_server:
	poetry run uvicorn application.application:app --port=5555

batch_prediction:
	poetry run python application/batch_prediction.py