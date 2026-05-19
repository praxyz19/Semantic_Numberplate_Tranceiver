from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import UnifiedPlateDataset
from .model import SemanticLPRNet, semantic_lpr_loss


def average_delta(server_state: dict[str, torch.Tensor], client_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: client_state[name] - server_state[name] for name in server_state}


def apply_async_update(server: SemanticLPRNet, client_state: dict[str, torch.Tensor], alpha: float) -> None:
    state = server.state_dict()
    for name in state:
        client_value = client_state[name].to(state[name].device)
        state[name] = state[name] + alpha * (client_value - state[name])
    server.load_state_dict(state)


def train_client(
    base_model: SemanticLPRNet,
    dataset_root: Path,
    client_id: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    client_count: int,
    ccpd_root: str | None,
    ccpd_splits: list[str],
    recon_weight: float = 1.0,
    text_weight: float = 1.0,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    model = copy.deepcopy(base_model).to(device)
    model.train()
    dataset = UnifiedPlateDataset(
        dataset_root,
        client_count=client_count,
        client_id=client_id,
        ccpd_root=ccpd_root,
        ccpd_splits=ccpd_splits,
        augment=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stats = {"loss": 0.0, "text_loss": 0.0, "recon_loss": 0.0, "steps": 0}

    for _ in range(local_epochs):
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            # SNR is randomised inside the model during training
            outputs = model(batch["image"])
            loss, detail = semantic_lpr_loss(outputs, batch, recon_weight=recon_weight, text_weight=text_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            for key in ("loss", "text_loss", "recon_loss"):
                stats[key] += detail[key]
            stats["steps"] += 1

    for key in ("loss", "text_loss", "recon_loss"):
        stats[key] /= max(stats["steps"], 1)
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}, stats


def run_async_fl(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset_root = Path(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    server = SemanticLPRNet(
        latent_dim=args.latent_dim,
        channel_dim=args.channel_dim,
        snr_range_db=(0.0, 20.0),
    ).to(device)
    client_versions = {client_id: 0 for client_id in range(args.clients)}
    history_path = output_dir / "history.json"
    history: list[dict] = []
    server_round = 0

    if getattr(args, "resume", None):
        resume_path = Path(args.resume)
        if resume_path.exists():
            checkpoint = torch.load(resume_path, map_location=device)
            model_state = checkpoint.get("model_state")
            if isinstance(model_state, dict):
                server.load_state_dict(model_state)
            server_round = int(checkpoint.get("round", 0))
            if history_path.exists():
                try:
                    history = json.loads(history_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    history = []
            print(f"Resuming from {resume_path} at round {server_round}")
        else:
            print(f"Resume checkpoint not found: {resume_path} — starting from scratch")

    print(f"Training on {device} | clients={args.clients} | rounds={args.rounds}")
    print(f"Model params: {sum(p.numel() for p in server.parameters()):,}")

    for round_idx in range(server_round, server_round + args.rounds):
        client_order = list(range(args.clients))
        random.shuffle(client_order)
        for client_id in client_order:
            start_version = server_round
            client_state, stats = train_client(
                server,
                dataset_root,
                client_id,
                args.local_epochs,
                args.batch_size,
                args.lr,
                device,
                client_count=args.clients,
                ccpd_root=args.ccpd_root,
                ccpd_splits=args.ccpd_splits,
                recon_weight=args.recon_weight,
                text_weight=args.text_weight,
            )
            staleness = max(0, server_round - client_versions[client_id])
            alpha = args.server_lr / (1.0 + args.staleness_decay * staleness)
            apply_async_update(server, client_state, alpha=alpha)
            server_round += 1
            client_versions[client_id] = server_round
            record = {
                "round": round_idx,
                "server_update": server_round,
                "client_id": client_id,
                "client_start_version": start_version,
                "staleness": staleness,
                "alpha": round(alpha, 5),
                **stats,
            }
            history.append(record)
            print(json.dumps(record))

        torch.save(
            {
                "model_state": server.state_dict(),
                "round": server_round,
                "args": vars(args),
            },
            output_dir / "semantic_lpr_async_fl.pt",
        )
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"=== Round {server_round} complete — checkpoint saved ===")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Asynchronous federated training for semantic licence plate communication.")
    parser.add_argument("--dataset", default="data")
    parser.add_argument("--ccpd-root", default=None)
    parser.add_argument("--ccpd-splits", default="train.txt,ccpd_blur.txt,ccpd_rotate.txt,ccpd_tilt.txt")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume from")
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--server-lr", type=float, default=0.65)
    parser.add_argument("--staleness-decay", type=float, default=0.5)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--channel-dim", type=int, default=64)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--text-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.ccpd_root:
        args.ccpd_splits = [item.strip() for item in str(args.ccpd_splits).split(",") if item.strip()]
    else:
        args.ccpd_splits = []
    run_async_fl(args)


if __name__ == "__main__":
    main()
