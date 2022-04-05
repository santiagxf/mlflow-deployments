# Effortless deployment with MLFlow

This repository contains several examples about how to create and train models with MLFlow to then seemlessly deploy them using built-on tools, both locally on your computer, on a custom target like Kubernetes or in a cloud provider like Azure Machine Learning.

For a detailed explanation about this samples see the posts of the series:
- [Effortless models deployment with MLFlow](https://santiagof.medium.com/effortless-models-deployment-with-mlflow-2b1b443ff157).
- [Effortless models deployment with MLFlow: Customizing inference](https://santiagof.medium.com/effortless-models-deployment-with-mlflow-customizing-inference-e880cd1c9bdd).
- [Effortless models deployment with MLFlow: Packaging models with multiple pieces](https://santiagof.medium.com/effortless-models-deployment-with-mlflow-models-with-multiple-pieces-f38443641c8d)

**The following samples are available:**
- [Cats vs Dogs classification model using FastAI](dogs-and-cats/fastai-dogs-and-cats.ipynb): A sample notebook that creates a computer vision classifier using transfer learning from a RestNet32 model. The framework used is FastAI.
- [Cats vs Dogs classifier with custom inference using FastAI and PyFunc](dogs-and-cats/fastai-dogs-and-cats-pyfunc.ipynb): A sample notebook that creates a computer vision classifier using transfer learning from a RestNet32 model, but runs a custom inference routine. The framework used is FastAI. This model shows how inference can be simply modified in MLFlow.
- [Recommendation system in MLFlow using a custom model flavor](event-recommender/event-recommender.ipynb): A sample notebook that creates a recommender system for users to attend to different events. The model recommends for a user the top 10 events they may be interested. The model uses the library `implicit` with the algorithmn Alternating Least Squares as in the paper [Applications of the Conjugate Gradient Method for Implicit Feedback Collaborative Filtering](https://dl.acm.org/doi/10.1145/2043932.2043987). This notebooks shows how you package custom models that combine multiple ML frameworks using MLFlow.

**Comming soon:**
- [Hate detection transformer for tweets in portuguese language created as a custom model in MLFlow](hatespeech-classifier/hate-pt-speech-mlflow.ipynb): A hate detection model based on transformers and BERT architecture to detect hate on tweets in portuguese language. This models show how to create custom models using the MLFlow specification while retaining easy deployment provided by MLFlow.

## Contributing

More than welcome! Open an issue to go over it!


```python

```
