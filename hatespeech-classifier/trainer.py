"""
Training routine for a language model using transformers
"""
import logging
import mlflow
from typing import Dict, Any
from types import SimpleNamespace
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from mlflow.types import DataType

from transformers import Trainer, TrainingArguments
from hatedetection.hate_detection_classifier import HateDetectionClassifier
from hatedetection.evaluation import compute_classification_metrics
from hatedetection.text_datasets import ClassificationDataset
from hatedetection.text_preparation import load_examples

def train_and_evaluate(input_dataset: str, 
                       eval_dataset: str,
                       params: SimpleNamespace):
    """
    Trains and evaluete the hate detection model

    Parameters
    ----------
    input_dataset: Union[str, PathLike]
        The path to the training dataset
    eval_dataset: Union[str, PathLike]
        The path to the evaluation dataset
    params: SimpleNamespace
        The training configuration for this task
    """
    classifier = HateDetectionClassifier()
    classifier.build(baseline=params.model.baseline)
    classifier.split_unique_words = params.data.preprocessing.split_unique_words
    classifier.split_seq_len = params.data.preprocessing.split_seq_len

    logging.info(f'[INFO] Loading training data from {input_dataset}')
    examples_train, labels_train = load_examples(input_dataset,
                                                 split_seq=True,
                                                 unique_words=params.data.preprocessing.split_unique_words,
                                                 seq_len = params.data.preprocessing.split_seq_len)
    if eval_dataset:
        logging.info(f'[INFO] Loading evaluation data from {eval_dataset}')
        examples_eval, labels_eval = load_examples(eval_dataset,
                                                   split_seq=True,
                                                   unique_words=params.data.preprocessing.split_unique_words,
                                                   seq_len = params.data.preprocessing.split_seq_len)
    else:
        logging.warning('[WARN] Evaluation will happen over the training dataset as evaluation \
                        dataset has not been provided.')
        examples_eval, labels_eval = examples_train, labels_train


    logging.info('[INFO] Building datasets for trainer object')
    train_dataset = ClassificationDataset(examples=examples_train,
                                          labels=labels_train,
                                          tokenizer=classifier.tokenizer)
    eval_dataset = ClassificationDataset(examples=examples_eval,
                                         labels=labels_eval,
                                         tokenizer=classifier.tokenizer)

    logging.info('[INFO] Reading training arguments')
    training_args = TrainingArguments(**vars(params.trainer))

    logging.info('[INFO] Building trainer')
    trainer = Trainer(
        model=classifier.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_classification_metrics
    )

    logging.info('[INFO] Training will start now')
    history = trainer.train()

    logging.info('[INFO] Evaluation will start now')
    evaluation_metrics = trainer.evaluate()

    logging.info('[INFO] Training completed. Persisting model and tokenizer.')
    saved_location=f"{params.model.output_dir}/{params.model.name}"
    classifier.save_pretrained(saved_location)
    
    logging.info('[INFO] Logging parameters.')
    mlflow.log_params(history.metrics)

    logging.info('[INFO] Logging metrics.')
    mlflow.log_metrics(dict(filter(lambda item: item[1] is not None, evaluation_metrics.items())))

    logging.info(f'[INFO] Logging model from path {saved_location}.')
    signature = ModelSignature(inputs=Schema([
                                    ColSpec(DataType.string, "text"),
                                ]),
                            outputs=Schema([
                                    ColSpec(DataType.integer, "hate"),
                                    ColSpec(DataType.double, "confidence"),
                                ]))
    mlflow.pyfunc.log_model("classifier", 
                        data_path=saved_location, 
                        code_path=["hatedetection"], 
                        loader_module="hatedetection.hate_detection_classifier", 
                        registered_model_name="hate-pt-speech", 
                        signature=signature)

    logging.info('[INFO] Train completed.')
