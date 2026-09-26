import os
import time
import pickle
import torch
import torch.nn as nn

from utils import to_dict
from losses import compute_loss


def _count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _grad_global_norm(model):
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += float(p.grad.detach().data.norm(2).item()) ** 2
    return total_sq ** 0.5


def train(cfg, model, train_dataloader):
    opt_cls = getattr(torch.optim, cfg.training.optimizer.name)
    optimizer = opt_cls(model.parameters(), **to_dict(cfg.training.optimizer.params))

    sch_cls = getattr(torch.optim.lr_scheduler, cfg.training.scheduler.name)
    scheduler = sch_cls(optimizer, **to_dict(cfg.training.scheduler.params))

    grad_clip = getattr(cfg.training, "grad_clip", None)
    if grad_clip is not None:
        grad_clip = float(grad_clip)

    loss_list = []
    runtime_list = []
    grad_norm_list = []
    lr_list = []
    component_history = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_params, trainable_params = _count_params(model)
    print(f"Model parameters: total={total_params:,} ({total_params/1e6:.2f}M), "
          f"trainable={trainable_params:,} ({trainable_params/1e6:.2f}M)")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        print(f"CUDA device: {torch.cuda.get_device_name(device)} "
              f"(visible index {torch.cuda.current_device()})")

    for epoch in range(cfg.training.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_grad_norm = 0.0
        epoch_components = {}
        num_batches = 0
        start_time = time.time()

        for data_sample in train_dataloader:
            x = data_sample.x.to(device).float()
            batch = data_sample.batch.to(device).float()
            particle_information = data_sample.particle_information.to(device).float()
            y = data_sample.y.to(device).float()

            optimizer.zero_grad()
            out = model(x, batch, particle_information)

            loss, components = compute_loss(out, y, x, batch, particle_information)

            loss.backward()

            batch_grad_norm = _grad_global_norm(model)
            epoch_grad_norm += batch_grad_norm

            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            epoch_loss += loss.item()
            for k, v in components.items():
                epoch_components[k] = epoch_components.get(k, 0.0) + float(v)
            num_batches += 1

        epoch_runtime = time.time() - start_time

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_grad_norm = epoch_grad_norm / max(num_batches, 1)
        current_lr = optimizer.param_groups[0]["lr"]

        loss_list.append(avg_loss)
        runtime_list.append(epoch_runtime)
        grad_norm_list.append(avg_grad_norm)
        lr_list.append(current_lr)
        for k, v in epoch_components.items():
            component_history.setdefault(k, []).append(v / max(num_batches, 1))

        scheduler.step()

        if epoch % cfg.training.save_freq == 0:
            model_save_path = os.path.join(cfg.training.exp_name, f"model.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch} to {model_save_path}")

        print(f"Epoch {epoch}/{cfg.training.num_epochs} - Loss: {avg_loss:.7f}, "
              f"GradNorm: {avg_grad_norm:.4f}, LR: {current_lr:.3e}, "
              f"Runtime: {epoch_runtime:.2f}s")

    if torch.cuda.is_available():
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_gb = peak_bytes / (1024 ** 3)
        print(f"Peak GPU memory used during training: {peak_gb:.3f} GB ({peak_bytes:,} bytes)")
    else:
        peak_bytes = 0
        peak_gb = 0.0

    results = {
        "loss": loss_list,
        "runtime": runtime_list,
        "grad_norm": grad_norm_list,
        "lr": lr_list,
        "loss_components": component_history,
        "num_params": total_params,
        "trainable_params": trainable_params,
        "peak_gpu_memory_bytes": peak_bytes,
        "peak_gpu_memory_gb": peak_gb,
    }
    txt_path = f"{cfg.training.exp_name}/results.txt"
    pkl_path = f"{cfg.training.exp_name}/results.pkl"

    model_save_path = os.path.join(cfg.training.exp_name, "model.pth")
    torch.save(model.state_dict(), model_save_path)

    with open(txt_path, "w") as f:
        f.write(f"Total parameters:    {total_params:,} ({total_params/1e6:.2f}M)\n")
        f.write(f"Trainable parameters:{trainable_params:,} ({trainable_params/1e6:.2f}M)\n")
        f.write(f"Peak GPU memory:     {peak_gb:.3f} GB\n\n")
        for epoch in range(len(loss_list)):
            f.write(
                f"Epoch {epoch+1}: Loss = {loss_list[epoch]:.7f}, "
                f"GradNorm = {grad_norm_list[epoch]:.4f}, "
                f"LR = {lr_list[epoch]:.3e}, "
                f"Runtime = {runtime_list[epoch]:.2f}s\n"
            )

    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Training completed. Results saved to {txt_path} and {pkl_path}")