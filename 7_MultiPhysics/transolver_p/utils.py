import os
import sys
import yaml
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


class Config(SimpleNamespace):
    pass


def _dict_to_namespace(d):
    if isinstance(d, dict):
        return Config(**{k: _dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_dict_to_namespace(v) for v in d]
    else:
        return d


def _deep_merge(base, override):
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _resolve_base(config):
    if "base_config" in config:
        base_path = config.pop("base_config")
        with open(base_path, "r") as f:
            base = yaml.safe_load(f) or {}
        base = _resolve_base(base)
        config = _deep_merge(base, config)
    return config


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    return _dict_to_namespace(_resolve_base(config))


BOUNDS = {
    "solid": (100.0, 1500.0),
    "flux": (-2000.0, 4000.0),
    "neutron": (0.0, 3.258),
    "fluid": ((100.0, 1200.0), (-50.0, 250.0), (-0.006, 0.006), (0.0, 0.6)),
}


def normalize(x, field):
    if field == "fluid":
        out = x.clone()
        for i in range(x.shape[1]):
            lo, hi = BOUNDS["fluid"][i]
            out[:, i] = (x[:, i] - lo) / (hi - lo) * 2 - 1
        return out
    if field == "neutron":
        lo, hi = BOUNDS["neutron"]
        return (torch.log(x + 1) - lo) / (hi - lo) * 2 - 1
    lo, hi = BOUNDS[field]
    return (x - lo) / (hi - lo) * 2 - 1


def renormalize(x, field):
    if field == "fluid":
        out = x.clone()
        for i in range(x.shape[1]):
            lo, hi = BOUNDS["fluid"][i]
            out[:, i] = (x[:, i] + 1) * 0.5 * (hi - lo) + lo
        return out
    if field == "neutron":
        lo, hi = BOUNDS["neutron"]
        return torch.exp((x + 1) * 0.5 * (hi - lo) + lo) - 1
    lo, hi = BOUNDS[field]
    return (x + 1) * 0.5 * (hi - lo) + lo


def assemble_cond(field, conds):
    if field == "neutron":
        bc, temp = conds
        bc = bc.expand(-1, -1, -1, -1, temp.shape[-1])
        return torch.cat((bc, temp), dim=1)
    if field == "solid":
        neu, fluid_if = conds
        fluid_if = fluid_if.expand(-1, -1, -1, -1, neu.shape[-1])
        return torch.cat((neu, fluid_if), dim=1)
    if field == "fluid":
        (flux,) = conds
        return flux.expand(-1, -1, -1, -1, 12)
    raise ValueError(field)


def load_decoupled(root, field, dataset="iter1", n_data_set=5000):
    folder = os.path.join(root, dataset)
    if field == "neutron":
        bc = normalize(torch.from_numpy(np.load(f"{folder}/bc_neu.npy")[:n_data_set]).float(), "neutron")
        fuel = normalize(torch.from_numpy(np.load(f"{folder}/fuel_neu.npy")[:n_data_set]).float(), "solid")
        fluid = normalize(torch.from_numpy(np.load(f"{folder}/fluid_neu.npy")[:n_data_set]).float(), "fluid")
        x = assemble_cond("neutron", [bc, torch.cat((fuel, fluid), dim=-1)])
        y = normalize(torch.from_numpy(np.load(f"{folder}/neu.npy")[:n_data_set]).float(), "neutron")
    elif field == "solid":
        neu = normalize(torch.from_numpy(np.load(f"{folder}/neu_fuel.npy")[:n_data_set]).float(), "neutron")
        fluid_if = normalize(torch.from_numpy(np.load(f"{folder}/fluid_fuel.npy")[:n_data_set]).float(), "fluid")
        x = assemble_cond("solid", [neu, fluid_if])
        y = normalize(torch.from_numpy(np.load(f"{folder}/fuel.npy")[:n_data_set]).float(), "solid")
    elif field == "fluid":
        flux = normalize(torch.from_numpy(np.load(f"{folder}/fuel_fluid.npy")[:n_data_set]).float(), "flux")
        x = assemble_cond("fluid", [flux])
        y = normalize(torch.from_numpy(np.load(f"{folder}/fluid.npy")[:n_data_set]).float(), "fluid")
    else:
        raise ValueError(field)
    return x, y


def _load_split(cfg, split):
    x, y = load_decoupled(cfg.data.dataset_root, cfg.data.field, cfg.data.dataset, cfg.data.n_data_set)
    gap = cfg.data.gap
    if split == "train":
        x, y = x[:-gap], y[:-gap]
        shuffle = True
    elif split == "valid":
        x, y = x[-gap:], y[-gap:]
        shuffle = False
    else:
        raise NotImplementedError
    print(f"x: {x.shape}")
    print(f"y: {y.shape}")
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=shuffle, pin_memory=True)


def load_train(cfg):
    return _load_split(cfg, "train")


def load_valid(cfg):
    return _load_split(cfg, "valid")


def load_couple(cfg):
    folder = os.path.join(cfg.data.dataset_root, "val")
    n = getattr(cfg.couple, "n_samples", None)
    bc = torch.from_numpy(np.load(f"{folder}/bc.npy")[:n]).float()
    neu = torch.from_numpy(np.load(f"{folder}/neu.npy")[:n]).float()
    fuel = torch.from_numpy(np.load(f"{folder}/fuel.npy")[:n]).float()
    fluid = torch.from_numpy(np.load(f"{folder}/fluid.npy")[:n]).float()
    return bc, neu, fuel, fluid
