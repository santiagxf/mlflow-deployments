# Effortless deployment with MLFlow

This repository contains several examples about how to create and train models with MLFlow to then seemlessly deploy them using built-on tools, both locally on your computer, on a custom target like Kubernetes or in a cloud provider like Azure Machine Learning.

For a detailed explanation about this samples see the posts of the series:
- [Effortless models deployment with MLFlow — An introduction](https://santiagof.medium.com/effortless-models-deployment-with-mlflow-2b1b443ff157).
- [Effortless models deployment with MLFlow — Customizing inference](https://santiagof.medium.com/effortless-models-deployment-with-mlflow-customizing-inference-e880cd1c9bdd).
- [Effortless models deployment with MLFlow — Packaging models with multiple pieces](https://santiagof.medium.com/effortless-models-deployment-with-mlflow-models-with-multiple-pieces-f38443641c8d)
- [Effortless models deployment with MLFlow — Packing a NLP product review classifier from HuggingFace](https://santiagof.medium.com/effortless-models-deployment-with-mlflow-packing-a-nlp-product-review-classifier-from-huggingface-13be2650333)
- [Effortless model deployment with MLflow — Stratified models (many models) for forecasting](https://santiagof.medium.com/effortless-model-deployment-with-mlflow-stratified-models-many-models-for-forecasting-a7b3d59cc5ee).

**The following samples are available:**
- [Cats vs Dogs classification model using FastAI](dogs-and-cats/fastai-dogs-and-cats.ipynb): A sample notebook that creates a computer vision classifier using transfer learning from a RestNet32 model. The framework used is FastAI.
- [Cats vs Dogs classifier with custom inference using FastAI and PyFunc](dogs-and-cats/fastai-dogs-and-cats-pyfunc.ipynb): A sample notebook that creates a computer vision classifier using transfer learning from a RestNet32 model, but runs a custom inference routine. The framework used is FastAI. This model shows how inference can be simply modified in MLFlow.
- [Recommendation system in MLFlow using a custom model flavor](event-recommender/event-recommender.ipynb): A sample notebook that creates a recommender system for users to attend to different events. The model recommends for a user the top 10 events they may be interested. The model uses the library `implicit` with the algorithmn Alternating Least Squares as in the paper [Applications of the Conjugate Gradient Method for Implicit Feedback Collaborative Filtering](https://dl.acm.org/doi/10.1145/2043932.2043987). This notebooks shows how you package custom models that combine multiple ML frameworks using MLFlow.
- [A product review classifier built with HuggingFace, trasnformers and Mlflow](transformer-classifier/bert-for-classification.ipynb): This notebook shows how to package a HuggingFace model using Mlfow to deploy easily anywhere. Concretely, the example shows how to deploy a classification model for product reviews in the scale of 1 to 5 stars.
- [Working with partitioned models (many models) in MLflow](partitioned-models/m5-forecasting-lightgbm-with-timeseries-splits.ipynb): This notebook solves the M5 forecasting problem from Kaggle where you are ask to predict the demand of different products across multiple stores in the US. Instead of training one single model to solve the problem, this example train 10 models, one for each store. An "aggregator" model is construct to serve predictions.

## Contributing

More than welcome! Open an issue to go over it!
