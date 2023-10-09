# MSDS - MLOps course - Foodformer <img src="./images/foodformer_logo.jpeg" alt="foodformer_logo" width="20"/>

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31011/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nico-usf/foodformer)

## Training

For model training, we harnessed the power of [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/) to fine-tune our model using the [Foodset 101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/), which is available through [TorchVision](https://pytorch.org/vision/stable/datasets.html#food101).

Our model is built upon the [VisionTransformer](https://huggingface.co/google/vit-base-patch16-224-in21k) architecture, known for its excellent performance in multi label image recognition tasks.

We conducted training over a span of 10 epochs, optimizing the model to achieve results through distributed data parallelism. To explore the training progress, metrics, and insights, please visit the [W&B dashboard](https://wandb.ai/maneel/Foodformer?workspace=user-maneel1995).



## Development

To setup this repo locally, create a virtual environment (e.g. with [PyEnv](https://github.com/pyenv/pyenv)):

```bash
brew install pyenv
pyenv init
pyenv install -s 3.10.10
pyenv virtualenv 3.10.10 foodformer
pyenv activate foodformer
pyenv local foodformer
```

then install the dependencies and pre-commit hooks:

```bash
pip install -r requirements.txt
pre-commit install
```

## Testing the API

You can use API platforms like Postman or Insomnia, the command-line tool `curl`.

- for the healthcheck endpoint: `curl http://localhost:8080`
- for a post endpoint called `predict`:

```bash
curl -X 'POST' \
  'http://35.88.15.190//predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@image.jpg;type=image/jpeg'
```

## Load testing Reports

For load testing via locust, follow the instructions.

```bash
cd load_testing
docker build --no-cache -t my-image:latest .
docker run -p 8890:8089 my-image:latest
```
Set the load testing parameters and take required screenshots. Screenshots from my test runs are as follows:-

![Image Alt Text](/load_testing/results/testreport.png)

Other screenshots can be found here 
1. [Image 1](/load_testing/results/testreport1.png)
2. [Image 2](/load_testing/results/testreport2.png)
3. [Image 3](/load_testing/results/testreport3.png)

## Grafana Dashboard

You can access the Grafana dashboard snapshot [here](https://snapshots.raintank.io/dashboard/snapshot/tfiUtgCyEUWr0yKDdKAdbsvU56tvglu6).

Explore valuable metrics and insights from our project on the dashboard.







