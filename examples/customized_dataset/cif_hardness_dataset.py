from graphormer.data import register_dataset
from torch_geometric.data import Dataset, Data
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
import torch
import csv
import os

from pymatgen.core import Structure


class CIFHardnessDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        cutoff: float,
        distance_bins: np.ndarray,
        max_neighbors: int,
    ):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.cutoff = cutoff
        self.distance_bins = distance_bins
        self.max_neighbors = max_neighbors
        self.samples = self._load_csv(self.csv_path)

    def _load_csv(self, csv_path: Path):
        base_dir = csv_path.parent
        samples = []
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"No header found in CSV file: {csv_path}")
            required = {"id", "cif_path", "hardness"}
            missing = required.difference(reader.fieldnames)
            if missing:
                missing_list = ", ".join(sorted(missing))
                raise ValueError(
                    f"Missing required columns in {csv_path}: {missing_list}"
                )
            for row in reader:
                if not any(row.values()):
                    continue
                cif_path = Path(row["cif_path"])
                if not cif_path.is_absolute():
                    cif_path = base_dir / cif_path
                split = row.get("split")
                split = split.strip().lower() if split else None
                samples.append(
                    {
                        "id": row["id"],
                        "cif_path": cif_path,
                        "hardness": float(row["hardness"]),
                        "split": split,
                    }
                )
        return samples

    def __len__(self):
        return len(self.samples)

    def _build_graph(self, structure: Structure):
        num_nodes = len(structure)
        node_features = torch.tensor(
            [[site.specie.number] for site in structure], dtype=torch.long
        )

        senders = []
        receivers = []
        distances = []

        for idx in range(num_nodes):
            neighbors = structure.get_neighbors(structure[idx], self.cutoff)
            neighbors = sorted(neighbors, key=lambda neighbor: neighbor.nn_distance)
            if self.max_neighbors > 0:
                neighbors = neighbors[: self.max_neighbors]
            for neighbor in neighbors:
                senders.append(idx)
                receivers.append(neighbor.index)
                distances.append(neighbor.nn_distance)

        if not senders:
            senders = list(range(num_nodes))
            receivers = list(range(num_nodes))
            distances = [0.0] * num_nodes

        edge_index = torch.tensor([senders, receivers], dtype=torch.long)
        distance_bins = np.digitize(np.array(distances), self.distance_bins)
        edge_attr = torch.tensor(distance_bins, dtype=torch.long)
        return node_features, edge_index, edge_attr

    def __getitem__(self, idx):
        sample = self.samples[idx]
        structure = Structure.from_file(sample["cif_path"])
        node_features, edge_index, edge_attr = self._build_graph(structure)
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([sample["hardness"]], dtype=torch.float),
        )
        data.sample_id = sample["id"]
        return data


@register_dataset("customized_cif_hardness")
def create_cif_hardness_dataset():
    csv_env = os.environ.get("CIF_HARDNESS_CSV")
    if not csv_env:
        raise ValueError("CIF_HARDNESS_CSV must be set to a hardness CSV path.")
    csv_path = Path(csv_env)
    cutoff = float(os.environ.get("CIF_HARDNESS_CUTOFF", 6.0))
    max_neighbors = int(os.environ.get("CIF_HARDNESS_MAX_NEIGHBORS", 32))
    bin_edges = os.environ.get("CIF_HARDNESS_DISTANCE_BINS", "2.0,3.0,4.0,5.0")
    distance_bins = np.array([float(value) for value in bin_edges.split(",")])
    if distance_bins.size == 0:
        raise ValueError("CIF_HARDNESS_DISTANCE_BINS must include at least one value.")
    if not np.all(np.diff(distance_bins) > 0):
        raise ValueError("CIF_HARDNESS_DISTANCE_BINS must be strictly increasing.")
    if max_neighbors < 0:
        raise ValueError("CIF_HARDNESS_MAX_NEIGHBORS must be >= 0.")

    dataset = CIFHardnessDataset(
        csv_path=csv_path,
        cutoff=cutoff,
        distance_bins=distance_bins,
        max_neighbors=max_neighbors,
    )
    split_values = [sample.get("split") for sample in dataset.samples]
    has_splits = any(split_values)
    if has_splits:
        if any(split is None for split in split_values):
            raise ValueError(
                "If any row defines split, all rows must include split values."
            )
        split_map = {"train": [], "valid": [], "val": [], "test": []}
        for idx, split in enumerate(split_values):
            if split not in split_map:
                raise ValueError(
                    "Split must be one of: train, valid, val, test."
                )
            split_map[split].append(idx)
        train_idx = np.array(split_map["train"])
        valid_idx = np.array(split_map["valid"] + split_map["val"])
        test_idx = np.array(split_map["test"])
        if train_idx.size == 0 or valid_idx.size == 0 or test_idx.size == 0:
            raise ValueError("Train/valid/test splits must each be non-empty.")
    else:
        num_graphs = len(dataset)
        train_valid_idx, test_idx = train_test_split(
            np.arange(num_graphs), test_size=num_graphs // 10, random_state=0
        )
        train_idx, valid_idx = train_test_split(
            train_valid_idx, test_size=num_graphs // 5, random_state=0
        )
    return {
        "dataset": dataset,
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "test_idx": test_idx,
        "source": "pyg",
    }
