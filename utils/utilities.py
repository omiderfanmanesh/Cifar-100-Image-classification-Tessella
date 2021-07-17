import os

import torch
from ignite.contrib.handlers.tensorboard_logger import *


def accuracy(y_true, y_pred):
    y_true = y_true.float()
    _, y_pred = torch.max(y_pred, dim=-1)
    return (y_pred.float() == y_true).float().mean()


def progress_bar(current, total, barLength=20):
    percent = float(current) * 100 / total
    arrow = '-' * int(percent / 100 * barLength - 1) + '>'
    spaces = ' ' * (barLength - len(arrow))

    print('Progress[%d]: [%s%s] %d %%' % (current, arrow, spaces, percent), end='\r')


def log_result(title, logger, engine, metrics):
    _avg_accuracy = metrics['accuracy']

    _precision = metrics['precision']
    _precision = torch.mean(_precision)

    _recall = metrics['recall']
    _recall = torch.mean(_recall)

    _f1 = metrics['f1']
    _f1 = torch.mean(_f1)

    _f1_micro = metrics['f1_micro']

    _avg_loss = metrics['ce_loss']

    logger.info(
        "{} Results - Epoch: {} Avg accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1 score: {:.3f}, "
        "f1_micro: {:.2f}, "
        "Avg Loss: {:.3f} ".format(title, engine.state.epoch, _avg_accuracy, _precision, _recall, _f1, _f1_micro,
                                   _avg_loss))


def tesnorboard(Events, model, optimizer, metrics, trainer, evaluator, log_dir):
    # Create a logger
    tb_logger = TensorboardLogger(log_dir=log_dir)

    # Attach the logger to the trainer to log training loss at each iteration
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training",
        output_transform=lambda loss: {"loss": loss}
    )

    # Attach the logger to the evaluator on the training dataset and log NLL, Accuracy metrics after each epoch
    # We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch
    # of the `trainer` instead of `evaluator`.
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="training",
        metric_names=metrics,
        global_step_transform=global_step_from_engine(trainer),
    )

    # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
    # each epoch. We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch of the
    # `trainer` instead of `evaluator`.
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names=metrics,
        global_step_transform=global_step_from_engine(trainer))

    # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
    tb_logger.attach_opt_params_handler(
        trainer,
        event_name=Events.ITERATION_STARTED,
        optimizer=optimizer,
        # param_name='lr'  # optional
    )

    # Attach the logger to the trainer to log model's weights norm after each iteration
    tb_logger.attach(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        log_handler=WeightsScalarHandler(model)
    )

    # Attach the logger to the trainer to log model's weights as a histogram after each epoch
    tb_logger.attach(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        log_handler=WeightsHistHandler(model)
    )

    # Attach the logger to the trainer to log model's gradients norm after each iteration
    tb_logger.attach(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        log_handler=GradsScalarHandler(model)
    )

    # Attach the logger to the trainer to log model's gradients as a histogram after each epoch
    tb_logger.attach(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        log_handler=GradsHistHandler(model)
    )

    # We need to close the logger when we are done
    tb_logger.close()


best_f1 = 0.
best_epoch = 1
best_epoch_file = ''


def get_saved_model_path(epoch, dir, model_name):
    return f'{dir}/Model_{model_name}_{epoch}.pth'


def save_best_epoch_only(epoch, dir, model_name, metrics, model):
    global best_f1
    global best_epoch
    global best_epoch_file
    best_f1 = 0. if epoch == 1 else best_f1
    best_epoch = 1 if epoch == 1 else best_epoch
    best_epoch_file = '' if epoch == 1 else best_epoch_file

    if metrics['accuracy'] > best_f1:
        prev_best_epoch_file = get_saved_model_path(epoch=best_epoch, dir=dir, model_name=model_name)
        if os.path.exists(prev_best_epoch_file):
            os.remove(prev_best_epoch_file)

        best_f1 = metrics['accuracy']
        best_epoch = epoch
        best_epoch_file = get_saved_model_path(epoch=best_epoch, dir=dir, model_name=model_name)
        print(f'\nEpoch: {best_epoch} - New best accuracy! accuracy: {best_f1}\n\n\n')
        torch.save(model.state_dict(), best_epoch_file)
