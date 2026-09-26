import os
import torch
from torch_geometric.loader import DataLoader

from shapenetcar import get_samples, get_datalist, GraphDataset

# In-process cache so load_train / load_valid / load_test don't rebuild the
# datalists (and coef_norm) three times.
_CACHE = {}


def _cache_key(cfg):
    return (cfg.data.dataset_root, cfg.data.fold_id)


def _build_datalists(cfg):
    """Build (or load from a cached .pth) the train/val datalists and coef_norm.

    Unlike the arrow-based loader, train and valid must be built together here:
    the normalization coefficients (coef_norm) are computed on the training
    folds and then applied to the held-out fold.
    """
    key = _cache_key(cfg)
    if key in _CACHE:
        return _CACHE[key]

    pth_file = None
    data_save_path = getattr(cfg.data, "data_save_path", None)
    if data_save_path:
        os.makedirs(data_save_path, exist_ok=True)
        pth_file = os.path.join(data_save_path, f"shapenetcar_fold{cfg.data.fold_id}.pth")

    if pth_file is not None and os.path.exists(pth_file):
        print(f"{os.path.basename(pth_file)} already exists. Load from {pth_file}")
        blob = torch.load(pth_file, weights_only=False)
        train_datalist = blob["train"]
        val_datalist = blob["val"]
        coef_norm = blob["coef_norm"]
    else:
        print(f".pth cache not found. Building dataset from {cfg.data.dataset_root}")
        samples = get_samples(cfg.data.dataset_root)
        assert 0 <= cfg.data.fold_id < len(samples), \
            f"fold_id must be in [0, {len(samples) - 1}], got {cfg.data.fold_id}"

        trainlst = []
        for i in range(len(samples)):
            if i == cfg.data.fold_id:
                continue
            trainlst += samples[i]
        vallst = samples[cfg.data.fold_id]

        train_datalist, coef_norm = get_datalist(
            cfg.data.dataset_root, trainlst, norm=True,
            savedir=cfg.data.save_dir, preprocessed=cfg.data.preprocessed,
        )
        val_datalist = get_datalist(
            cfg.data.dataset_root, vallst, coef_norm=coef_norm,
            savedir=cfg.data.save_dir, preprocessed=cfg.data.preprocessed,
        )

        print(f"len(train_datalist): {len(train_datalist)}")
        print(f"len(val_datalist): {len(val_datalist)}")

        if pth_file is not None:
            torch.save(
                {"train": train_datalist, "val": val_datalist, "coef_norm": coef_norm},
                pth_file,
            )
            print(f"Saved preprocessed dataset to {pth_file}")

    _CACHE[key] = (train_datalist, val_datalist, coef_norm)
    return _CACHE[key]


def _make_loader(cfg, datalist, batch_size, shuffle, drop_last=False):
    dataset = GraphDataset(
        datalist,
        use_height=getattr(cfg.data, "use_height", False),
        use_cfd_mesh=cfg.data.cfd_mesh,
        r=getattr(cfg.data, "r", None),
    )
    
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def load_train(cfg):
    train_datalist, _, _ = _build_datalists(cfg)
    return _make_loader(
        cfg, train_datalist,
        batch_size=cfg.training.batch_size, shuffle=True, drop_last=True,
    )


def load_valid(cfg):
    _, val_datalist, _ = _build_datalists(cfg)
    return _make_loader(cfg, val_datalist, batch_size=1, shuffle=False)


def load_test(cfg):
    return load_valid(cfg)


def get_coef_norm(cfg):
    """Return (mean_in, std_in, mean_out, std_out) computed on the training folds.

    Needed at eval time to de-normalize predictions (e.g. for drag/physical metrics).
    """
    _, _, coef_norm = _build_datalists(cfg)
    return coef_norm