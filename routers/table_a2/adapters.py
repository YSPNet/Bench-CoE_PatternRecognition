from __future__ import annotations

import importlib
import random
from pathlib import Path

import numpy as np

from .config import END_TO_END, FEATURE_METHODS, ExperimentError, lambda_value


CLASSES = {
    "oracle": ("routers.baselines.oracle", "Oracle"),
    "strongest": ("routers.baselines.strongest_global", "StrongestGlobal"),
    "cheapest": ("routers.baselines.cheapest_global", "CheapestGlobal"),
    "knn": ("routers.knn.router", "KNNRouter"),
    "prknn": ("routers.prknn.router", "PRKNNRouter"),
    "ovr": ("routers.ovr.router", "OVRRouter"),
    "kmeans": ("routers.kmeans.router", "KMeansRouter"),
    "linear": ("routers.linear.router", "LinearRouter"),
    "mlp": ("routers.mlp.router", "MLPRouter"),
    "cosinecls": ("routers.cosinecls.router", "CosineCLSRRouter"),
    "routerdc": ("routers.routerdc.router", "RouterDCRouter"),
    "zooter": ("routers.zooter.router", "ZooterRouter"),
    "vlc": ("routers.vlc.router", "VLCRouter"),
}


def router_class(method: str):
    module, name = CLASSES[method]
    return getattr(importlib.import_module(module), name)


def create_router(method: str, config: dict, lam: str, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    parameters = dict(config["method_parameters"].get(method, {}))
    protected = {"device", "use_soft_labels", "train_lambda", "enable_monitoring", "monitor_metric",
                 "random_state", "max_epochs", "max_iter", "learning_rate", "learning_rate_init",
                 "batch_size", "weight_decay", "patience", "use_pytorch"}
    if protected & parameters.keys():
        raise ExperimentError(f"use shared training/runtime fields instead of overriding {sorted(protected & parameters.keys())}")
    if method in FEATURE_METHODS:
        parameters.update({key: config["features"][key] for key in
                           ("fusion_method", "text_weight", "text_encoder", "vision_encoder")})
    if method in ("linear", "mlp") or method in END_TO_END:
        import torch
        if config["runtime"]["device"] == "cuda" and not torch.cuda.is_available():
            raise ExperimentError("CUDA is unavailable; select --device cpu explicitly")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        parameters.update(device=config["runtime"]["device"], use_soft_labels=True,
                          train_lambda=lambda_value(lam), enable_monitoring=True,
                          monitor_metric="rank_score", patience=config["training"]["patience"],
                          verbose=config["runtime"]["verbose"])
    if method in ("linear", "mlp"):
        parameters["max_iter"] = config["training"]["max_epochs"]
        if method == "mlp":
            parameters["random_state"] = seed
            parameters["hidden_sizes"] = tuple(parameters.get("hidden_sizes", [2048, 1024]))
            parameters["learning_rate_init"] = config["training"]["learning_rate"]
    if method in END_TO_END:
        parameters.update(config["training"])
        if method == "vlc":
            parameters["random_state"] = seed
    if method in ("ovr", "kmeans", "prknn"):
        parameters.setdefault("verbose", config["runtime"]["verbose"])
    return router_class(method)(**parameters)


def predictions(router, method: str, meta, text=None, vision=None, batch_size=16) -> np.ndarray:
    if method in FEATURE_METHODS:
        result = router.predict(X_text=text, X_vision=vision)
    elif method in END_TO_END:
        result = router.predict(meta=meta, batch_size=batch_size, num_workers=0)
    else:
        result = router.predict(meta=meta)
    return np.asarray(result)


def fit_classifier(router, method: str, config: dict, y, c, text, vision,
                   y_dev, c_dev, text_dev, vision_dev, models, bounds, output: Path):
    """Reuse native feature networks/targets with actual soft CE on CPU or GPU.

    Native CPU Linear silently argmaxes soft labels. This adapter keeps K expert
    columns, including at infinity, and uses the shared minibatch AdamW protocol.
    """
    import torch
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset
    from routers.utils.fusion import fuse_embeddings
    from routers.utils.rank_score import rank_score

    training = config["training"]
    labels, valid = router._derive_labels(y, c, use_soft_labels=True,
                                         soft_label_lambda=router.train_lambda)
    if not len(valid):
        raise ExperimentError("no train_seen rows have a correct expert")
    targets = labels[valid]
    features = config["features"]
    x = fuse_embeddings(text, vision, features["fusion_method"], features["text_weight"])
    xd = fuse_embeddings(text_dev, vision_dev, features["fusion_method"], features["text_weight"])
    router.scaler = StandardScaler().fit(x[valid])
    x = router.scaler.transform(x[valid]).astype(np.float32)
    xd = router.scaler.transform(xd).astype(np.float32)
    router.model_mapping = dict(enumerate(models))
    router.reverse_mapping = {name: column for column, name in router.model_mapping.items()}
    router.costs = c.mean(axis=0)
    router._last_X_shape = x.shape
    # StandardScaler stores CPU numpy arrays even when the network uses CUDA.
    router.use_gpu = False
    if method == "linear":
        from routers.linear.router import PyTorchLogisticRegression
        model = PyTorchLogisticRegression(x.shape[1], len(models), router.penalty, router.C)
    else:
        from routers.mlp.router import PyTorchMLP
        model = PyTorchMLP(x.shape[1], router.hidden_sizes, len(models), router.activation)
    router.pytorch_model = model.to(router.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                 weight_decay=training["weight_decay"])
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(targets)),
                        batch_size=training["batch_size"], shuffle=True, num_workers=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training["max_epochs"])
    dev_tensor = torch.from_numpy(xd).to(router.device)
    best_score, best_state, stale, best_epoch, history = -float("inf"), None, 0, None, []
    for epoch in range(1, training["max_epochs"] + 1):
        model.train()
        loss_sum = 0.0
        for xb, target in loader:
            xb, target = xb.to(router.device), target.to(router.device)
            optimizer.zero_grad()
            loss = -(target * model(xb).log_softmax(dim=-1)).sum(dim=-1).mean()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(xb)
        scheduler.step()
        model.eval()
        with torch.no_grad():
            choice = model(dev_tensor).argmax(dim=1).cpu().numpy()
        accuracy = float(y_dev[np.arange(len(choice)), choice].mean())
        avg_cost = float(c_dev[np.arange(len(choice)), choice].mean())
        score = rank_score(accuracy, avg_cost, *bounds, beta=config["rank_score_beta"])
        history.append({"epoch": epoch, "train_soft_ce": loss_sum / len(x),
                        "dev_accuracy": accuracy, "dev_avg_cost": avg_cost, "dev_rank_score": score})
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= training["patience"]:
            break
    model.load_state_dict(best_state)
    from .data import atomic_json
    atomic_json(output / "training_history.json", {"optimizer": "AdamW", "loss": "soft_cross_entropy",
                                                 "scheduler": "cosine", "best_epoch": best_epoch,
                                                 "selection_view": "dev_seen", "epochs": history})


def load_router(method: str, path: Path, config: dict):
    cls = router_class(method)
    if method in ("cosinecls", "routerdc", "zooter"):
        return cls.load(str(path), device=config["runtime"]["device"])
    router = cls.load(str(path))
    if method in ("linear", "mlp", "vlc"):
        router.device = config["runtime"]["device"]
        if method in ("linear", "mlp"):
            router.use_gpu = False
        network = getattr(router, "pytorch_model", None) if method != "vlc" else router.model
        if network is not None:
            network.to(router.device)
    return router
