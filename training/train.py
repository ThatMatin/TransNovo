import torch
import torch.nn as nn
import time
from typing import Optional
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils import data
from tqdm.auto import tqdm

import training
from logger import setup_logger
from interrupt import InterruptHandler
from modules.transformer import TransNovo
from training import mean_batch_acc

logger = setup_logger(__name__)

def train_step(model: nn.Module,
               optimizer: torch.optim.Optimizer,
               loss_fn: nn.Module,
               train_dl: data.DataLoader,
               scheduler: Optional[LambdaLR],
               interruptHandler: InterruptHandler):

    result_matrix = torch.zeros((len(train_dl), 3))

    model.train()
    for i, (X,Y,Ch,P) in tqdm(enumerate(train_dl),
                         total=len(train_dl),
                         desc="over training set"):
        logits = model(X, Y, Ch, P)

        # NOTE: For intensities with large values this returns nan
        # one fix is to discretize
        if torch.isnan(logits).any().item():
            logger.debug("NAN logit")
            del X, Y, Ch, P, logits
            torch.cuda.empty_cache()
            continue

        optimizer.zero_grad(True)

        tgt_output = Y[:, 1:]
        logits_flat = logits.transpose(-2, -1)

        loss = loss_fn(logits_flat, tgt_output)

        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        result_matrix[i, 0] = loss.detach()
        result_matrix[i, 1] = mean_batch_acc(logits.detach(), tgt_output.detach())
        result_matrix[i, 2] = model.grad_norms_mean()
        
        del X, Y, Ch, P, loss, logits
        torch.cuda.empty_cache()
        if interruptHandler.is_interrupted():
            break


    return result_matrix


def test_step(model: nn.Module,
              loss_fn: nn.Module,
              test_dl: data.DataLoader,
              interruptHandler: InterruptHandler):

    with torch.inference_mode():
        result_matrix = torch.zeros((len(test_dl), 2))
        model.eval()

        for i, (X,Y, Ch, P) in tqdm(enumerate(test_dl),
                             total=len(test_dl),
                             desc="over test set"):
            logits = model(X, Y, Ch, P)
            if torch.isnan(logits).any().item():
                logger.debug("NAN logit (test)")
                del X, Y, Ch, P, logits
                torch.cuda.empty_cache()
                continue

            tgt_output = Y[:, 1:]
            logits_flat = logits.transpose(-2, -1)
            loss = loss_fn(logits_flat, tgt_output)

            result_matrix[i, 0] = loss
            result_matrix[i, 1] = mean_batch_acc(logits, tgt_output)

            del X, Y, Ch, P
            torch.cuda.empty_cache()

            if interruptHandler.is_interrupted():
                break


    return result_matrix


def update_lr(optimizer: torch.optim.Optimizer, lr: float):
    for p in optimizer.param_groups:
        p['lr'] = lr

def init_adam(model: TransNovo):
    p = model.hyper_params
    betas = p.optimizer_adam_betas
    eps = p.optimizer_adam_eps
    lr = p.new_learning_rate
    steps = int(p.data_point_count/p.batch_size)
    total_steps = (p.n_epochs + p.n_epochs_sofar) * steps
    warmup_steps = p.warmup_steps

    print(f"Optimizer init:\n\ttotal steps: {total_steps}\n\twarmup steps: {warmup_steps}")

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))).item()

    a = Adam(model.parameters(), lr, betas, eps, 1e-5)
    scheduler = LambdaLR(a, lr_lambda)
    if len(model.hyper_params.optimizer_state_dict) != 0:
        a.load_state_dict(model.hyper_params.optimizer_state_dict)
    return a, scheduler


def train_loop(model: TransNovo, optimizer, loss_fn, train_dl, test_dl, interruptHandler: InterruptHandler, scheduler: Optional[LambdaLR] = None):
    rm_idx = 0
    p = model.hyper_params
    lr = optimizer.param_groups[0]['lr']

    start_epoch = p.n_epochs_sofar
    end_epoch = p.n_epochs
    ranger = tqdm(range(start_epoch, end_epoch), initial=start_epoch, total=end_epoch, desc="Epochs")

    s_time = time.time()
    # TODO: Find a proper place for it
    view_rate = 1 # p.n_epochs // 10

    # loss, acc, grad norms mean
    train_result_matrix = torch.zeros((p.n_epochs, len(train_dl), 3))
    # loss, acc
    test_result_matrix = torch.zeros((p.n_epochs, len(test_dl), 2))

    for epoch in ranger:

        rm_idx = epoch - start_epoch

        train_result_matrix[rm_idx] = training.train_step(model, optimizer, loss_fn, train_dl, scheduler, interruptHandler)
        test_result_matrix[rm_idx] = training.test_step(model, loss_fn, test_dl, interruptHandler)
        # Update learning rate
        lr = optimizer.param_groups[0]['lr']
        p.last_learning_rate = lr

        # calculate batch loss
        train_epoch_loss_tensor = train_result_matrix[rm_idx, :, 0]
        tr_l = train_epoch_loss_tensor[train_epoch_loss_tensor != 0].mean().item()

        train_epoch_acc_tensor = train_result_matrix[rm_idx, :, 1]
        tr_a = train_epoch_acc_tensor[train_epoch_acc_tensor != 0].mean().item()

        tr_epoch_grads = train_result_matrix[rm_idx, :, 2].mean()

        test_epoch_loss_tensor = test_result_matrix[rm_idx, :, 0]
        te_l = test_epoch_loss_tensor[test_epoch_loss_tensor != 0].mean().item()

        test_epoch_acc_tensor = test_result_matrix[rm_idx, :, 1]
        te_a = test_epoch_acc_tensor[test_epoch_acc_tensor != 0].mean().item()

        if epoch % view_rate == 0:
            t = "\n\tEpoch |    lr    | train loss | train acc | train norms mean | test loss | test acc"
            t += f"\n\t{epoch:>6}|{lr:>8.8f}|{tr_l:>12.6f}|{tr_a:>11.6f}|{tr_epoch_grads:>18.6f}|{te_l:>11.6f}|{te_a:>10.6f}\n"
            print(t)

        # TODO: Save after evey N iterations
        if interruptHandler.is_interrupted():
            break

    print(f"training time: {time.time() - s_time: 0.1f}s")
    model.finish_training(rm_idx,train_result_matrix,test_result_matrix,optimizer)
