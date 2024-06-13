import torch
import torch.nn as nn
import time
from torch.optim import Adam
from tqdm.auto import tqdm
from torch.utils import data

import train
from interrupt import InterruptHandler
from modules import TransNovo


def train_step(model: nn.Module,
               optimizer: torch.optim.Optimizer,
               loss_fn: nn.Module,
               train_dl: data.DataLoader):

    result_matrix = torch.zeros((len(train_dl), 2))

    model.train()
    for i, (X,Y) in enumerate(train_dl):
        logits = model(X, Y)

        optimizer.zero_grad(True)

        tgt_output = Y[:, 1:]
        logits_flat = logits.transpose(-2, -1)

        loss = loss_fn(logits_flat, tgt_output)

        loss.backward()
        optimizer.step()

        result_matrix[i, 0] = loss
        result_matrix[i, 1] = model.grad_norms_mean()

    return result_matrix


def test_step(model: nn.Module,
              loss_fn: nn.Module,
              test_dl: data.DataLoader):

    with torch.inference_mode():
        result_matrix = torch.zeros((len(test_dl), 2))
        model.eval()

        for i, (X,Y) in enumerate(test_dl):
            logits = model(X, Y)

            tgt_output = Y[:, 1:]
            logits_flat = logits.transpose(-2, -1)
            loss = loss_fn(logits_flat, tgt_output)

            result_matrix[i, 0] = loss
            result_matrix[i, 1] = model.grad_norms_mean()

    return result_matrix


def update_lr(optimizer: torch.optim.Optimizer, lr: float):
    for p in optimizer.param_groups:
        p['lr'] = lr

def init_adam(model: TransNovo):
    p = model.hyper_params
    d_model = p.d_model
    warmup = p.warmup_steps
    betas = p.optimizer_adam_betas
    eps = p.optimizer_adam_eps
    step = 1 if p.n_epochs_sofar == 0 else p.n_epochs_sofar
    lr = p.learning_rate(step, d_model, warmup)
    return Adam(model.parameters(), lr, betas, eps)


def train_loop(model: TransNovo, optimizer, loss_fn, train_dl, test_dl, interruptHandler: InterruptHandler):
    epoch = 0
    p = model.hyper_params

    start_epoch = p.n_epochs_sofar
    end_epochs = p.n_epochs_sofar + p.n_epochs

    s_time = time.time()
    # TODO: Find a proper place for it
    view_rate = 1 # p.n_epochs // 10

    train_result_matrix = torch.zeros((p.n_epochs, len(train_dl), 2))
    test_result_matrix = torch.zeros((p.n_epochs, len(test_dl), 2))

    for epoch in tqdm(range(start_epoch, end_epochs)):
        rm_idx = epoch - start_epoch

        train_result_matrix[rm_idx] = train.train_step(model, optimizer, loss_fn, train_dl)
        test_result_matrix[rm_idx] = train.test_step(model, loss_fn, test_dl)

        # Update learning rate
        lr = p.learning_rate(epoch + 1, p.d_model, p.warmup_steps)
        train.update_lr(optimizer, lr)

        if epoch % view_rate == 0:
            print(f"epoch: {epoch} | lr: {lr} | train loss: "
                  f"{train_result_matrix[rm_idx, :, 0].mean().item():.4f} "
                  f"| test loss: {train_result_matrix[rm_idx, :, 0].mean().item():.4f}")

        if interruptHandler.is_interrupted():
            break

    print(f"training time: {time.time() - s_time: 0.1f}s")
    model.finish_training(p.n_epochs, train_result_matrix,test_result_matrix,optimizer)
