#!/usr/bin/env python
import csv
import logging
from pathlib import Path

import torch
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar


def write_predictions(output_path: Path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "true", "prediction"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_split(cfg, args, task, model, split: str, device: torch.device):
    task.load_dataset(split)
    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=cfg.dataset.max_tokens_valid,
        max_sentences=cfg.dataset.batch_size_valid,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            model.max_positions(),
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_workers=cfg.dataset.num_workers,
        epoch=0,
        data_buffer_size=cfg.dataset.data_buffer_size,
        disable_iterator_cache=False,
    )
    itr = batch_iterator.next_epoch_itr(shuffle=False, set_dataset_epoch=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    rows = []
    model.eval()
    with torch.no_grad():
        for sample in progress:
            if device.type == "cuda":
                sample = utils.move_to_cuda(sample)
            outputs = model(**sample["net_input"])[:, 0, :].reshape(-1)
            targets = sample["target"].reshape(-1)[: outputs.shape[0]]
            sample_ids = sample["net_input"]["batched_data"].get("sample_id")
            if sample_ids is None:
                raise ValueError(
                    "sample_id is missing from batched data. Ensure your dataset "
                    "provides a sample_id attribute."
                )
            for sample_id, true_value, prediction in zip(
                sample_ids,
                targets.detach().cpu().tolist(),
                outputs.detach().cpu().tolist(),
            ):
                rows.append(
                    {
                        "id": sample_id,
                        "true": true_value,
                        "prediction": prediction,
                    }
                )
    return rows


def main():
    parser = options.get_training_parser()
    parser.add_argument("--checkpoint-path", required=True, type=str)
    parser.add_argument(
        "--splits",
        type=str,
        default="train,valid,test",
        help="comma-separated list of splits to export",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = options.parse_args_and_arch(parser, modify_parser=None)
    cfg = convert_namespace_to_omegaconf(args)

    logger = logging.getLogger(__name__)
    utils.set_torch_seed(cfg.common.seed)

    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)

    state = checkpoint_utils.load_checkpoint_to_cpu(args.checkpoint_path)
    model.load_state_dict(state["model"], strict=True, model_cfg=cfg.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        logger.info("Exporting predictions for split: %s", split)
        rows = run_split(cfg, args, task, model, split, device)
        output_path = args.output_dir / f"{split}_predictions.csv"
        write_predictions(output_path, rows)


if __name__ == "__main__":
    main()
