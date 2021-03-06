import logging

import torch
from ignite.engine import Events
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy, Fbeta
from ignite.metrics.precision import Precision
from ignite.metrics.recall import Recall

from metrics.f1score import F1Score


def inference(
        cfg,
        model,
        val_loader
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("template_model.inference")
    logger.info("Start inferencing")

    precision = Precision(average=False)
    recall = Recall(average=False)
    F1 = Fbeta(beta=1.0, average=False, precision=precision, recall=recall)
    metrics = {'accuracy': Accuracy(),
               'precision': precision,
               'recall': recall,
               'f1_micro': F1Score(),
               'f1': F1}
    evaluator = create_supervised_evaluator(model,
                                            metrics=metrics,
                                            device=device)

    # adding handlers using `evaluator.on` decorator API
    @evaluator.on(Events.EPOCH_COMPLETED)
    def print_validation_results(engine):
        metrics = evaluator.state.metrics


        _avg_accuracy = metrics['accuracy']

        _precision = metrics['precision']
        _precision = torch.mean(_precision)

        _recall = metrics['recall']
        _recall = torch.mean(_recall)

        _f1 = metrics['f1']
        _f1 = torch.mean(_f1)

        _f1_micro = metrics['f1_micro']
        logger.info(
            "Test Results - Epoch: {} Avg accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1 score: {:.3f}, f1_micro: {:.2f}".format(
                engine.state.epoch, _avg_accuracy, _precision, _recall, _f1, _f1_micro))

    evaluator.run(val_loader)
