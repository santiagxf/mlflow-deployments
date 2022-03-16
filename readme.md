# Effortless deployment with MLFlow

This repository contains several examples about how to create and train models with MLFlow to then seemlessly deploy them using built-on tools, both locally on your computer, on a custom target like Kubernetes or in a cloud provider like Azure Machine Learning.

For a detailed explanation about this samples see my post [Effortless models deployment with MLFlow](https://santiagof.medium.com/effortless-models-deployment-with-mlflow-2b1b443ff157).

**The following samples are available:**
- [Cats vs Dogs classification model using FastAI](fastai-dogs-and-cats.ipynb): A sample notebook that creates a computer vision classifier using transfer learning from a RestNet32 model. The framework used is FastAI.
- [Cats vs Dogs classifier with custom inference using FastAI and PyFunc](fastai-dogs-and-cats-pyfunc.ipynb): A sample notebook that creates a computer vision classifier using transfer learning from a RestNet32 model, but runs a custom inference routine. The framework used is FastAI. This model shows how inference can be simply modified in MLFlow.

**Comming soon:**
- [Hate detection transformer for tweets in portuguese language created as a custom model in MLFlow](hate-pt-speech-mlflow.ipynb): A hate detection model based on transformers and BERT architecture to detect hate on tweets in portuguese language. This models show how to create custom models using the MLFlow specification while retaining easy deployment provided by MLFlow.