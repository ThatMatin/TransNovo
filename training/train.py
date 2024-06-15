import torch
import torch.nn as nn
import time
from torch.optim import Adam
from tqdm.auto import tqdm
from torch.utils import data
from tqdm.auto import tqdm

import training
from interrupt import InterruptHandler
from modules.transformer import TransNovo
from training import mean_batch_acc


def train_step(model: nn.Module,
               optimizer: torch.optim.Optimizer,
               loss_fn: nn.Module,
               train_dl: data.DataLoader,
               interruptHandler: InterruptHandler):

    result_matrix = torch.zeros((len(train_dl), 3))

    model.train()
    for i, (X,Y) in tqdm(enumerate(train_dl),
                         total=len(train_dl),
                         desc="over training set"):
        logits = model(X, Y)

        optimizer.zero_grad(True)

        tgt_output = Y[:, 1:]
        logits_flat = logits.transpose(-2, -1)

        loss = loss_fn(logits_flat, tgt_output)

        loss.backward()
        optimizer.step()

        result_matrix[i, 0] = loss.detach()
        result_matrix[i, 1] = mean_batch_acc(logits, tgt_output)
        result_matrix[i, 2] = model.grad_norms_mean()

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

        for i, (X,Y) in tqdm(enumerate(test_dl),
                             total=len(test_dl),
                             desc="over test set"):
            logits = model(X, Y)

            tgt_output = Y[:, 1:]
            logits_flat = logits.transpose(-2, -1)
            loss = loss_fn(logits_flat, tgt_output)

            result_matrix[i, 0] = loss
            result_matrix[i, 1] = mean_batch_acc(logits, tgt_output)

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
    lr = p.learning_rate

    a = Adam(model.parameters(), lr, betas, eps)
    if len(model.hyper_params.optimizer_state_dict) != 0:
        a.load_state_dict(model.hyper_params.optimizer_state_dict)
    return a


def train_loop(model: TransNovo, optimizer, loss_fn, train_dl, test_dl, interruptHandler: InterruptHandler):
    rm_idx = 0
    lr = 0
    p = model.hyper_params

    start_epoch = p.n_epochs_sofar
    end_epochs = p.n_epochs_sofar + p.n_epochs

    s_time = time.time()
    # TODO: Find a proper place for it
    view_rate = 1 # p.n_epochs // 10

    # loss, acc, grad norms mean
    train_result_matrix = torch.zeros((p.n_epochs, len(train_dl), 3))
    # loss, acc
    test_result_matrix = torch.zeros((p.n_epochs, len(test_dl), 2))

    for epoch in tqdm(range(start_epoch, end_epochs), desc="Epochs"):
        rm_idx = epoch - start_epoch

        train_result_matrix[rm_idx] = training.train_step(model, optimizer, loss_fn, train_dl, interruptHandler)
        test_result_matrix[rm_idx] = training.test_step(model, loss_fn, test_dl, interruptHandler)

        # Update learning rate
        lr = p.learning_rate

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
            t = "\n\tEpoch |  lr  | train loss | train acc | train norms mean | test loss | test acc"
            t += f"\n\t{epoch:>6}|{lr:>6.6}|{tr_l:>12.6f}|{tr_a:>11.6f}|{tr_epoch_grads:>18.6f}|{te_l:>11.6f}|{te_a:>10.6f}\n"
            print(t)

        # TODO: Save after evey N iterations
        if interruptHandler.is_interrupted():
            break

    print(f"training time: {time.time() - s_time: 0.1f}s")
    model.finish_training(rm_idx,train_result_matrix,test_result_matrix,optimizer)
