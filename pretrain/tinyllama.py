import glob
import math
import sys
import time
from pathlib import Path
from typing import  Tuple
import math
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader
from functools import partial
import os
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
# from apex.optimizers import FusedAdam #torch optimizer has a cuda backend, which is faster actually
from lit_gpt.model import GPT, Block, Config
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops
from lit_gpt.utils import chunked_cross_entropy, get_default_supported_precision, num_parameters, step_csv_logger
from pytorch_lightning.loggers import WandbLogger
from lit_gpt import FusedCrossEntropyLoss
import random
import yaml
from types import SimpleNamespace



def setup(
    training_config: str = "experiments/tinyllama.yaml"
) -> None:
    precision = get_default_supported_precision(training=True)

    with open(training_config, 'r') as file:
        training_config = yaml.safe_load(file)
        training_config = SimpleNamespace(**training_config)
    logger = step_csv_logger("out", training_config.name, flush_logs_every_n_steps=training_config.log_step_interval * training_config.global_batch_size // training_config.num_of_devices // training_config.micro_batch_size)
    wandb_logger = WandbLogger(name=training_config.name, project="scaling_law")
    if training_config.num_of_devices > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block} if training_config.use_activation_checkpoint else None,
            sharding_strategy="SHARD_GRAD_OP",
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"
    fabric = L.Fabric(devices=training_config.num_of_devices, strategy=strategy, precision=precision, loggers=[logger, wandb_logger])
    fabric.print(training_config)
    
    training_config.out_dir = "out/" + training_config.name
    if fabric.global_rank == 0:
        os.makedirs(training_config.out_dir,  exist_ok=True)
        # save training_config to out_dir
        with open(training_config.out_dir+ '/training_config.yaml', 'w') as file:
            yaml.dump(vars(training_config), file)
    training_config.save_step_list = get_eval_step(training_config.max_step)
    training_config.eval_step_list = get_eval_step(training_config.max_step)
    training_config.batch_size = training_config.global_batch_size // training_config.num_of_devices
    training_config.gradient_accumulation_steps = training_config.batch_size // training_config.micro_batch_size
    assert training_config.gradient_accumulation_steps > 0
    training_config.warmup_iters = training_config.warmup_steps * training_config.gradient_accumulation_steps
    training_config.max_iters = training_config.max_step * training_config.gradient_accumulation_steps
    training_config.lr_decay_iters = training_config.max_iters
    training_config.log_iter_interval = training_config.log_step_interval * training_config.gradient_accumulation_steps
    training_config.out_dir = Path(training_config.out_dir)
    training_config.pretrained_path = Path(training_config.pretrained_path) if training_config.pretrained_path else None
    training_config.train_data_dir = Path(training_config.train_data_dir)
    training_config.val_data_dir = Path(training_config.val_data_dir)
    fabric.launch(main, training_config)
    # main(fabric, training_config)

def main(fabric, training_config):
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=training_config.log_iter_interval)

    model_config = Config.from_name(training_config.model_name)
    training_config.model_config = model_config
    train_dataloader, val_dataloader = create_dataloaders(
        fabric=fabric, training_config=training_config
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    fabric.seed_everything(3407)  # same seed for every process to init model (FSDP)

    fabric.print(f"Loading model with {model_config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = GPT(model_config)
        model.apply(partial(model._init_weights ,n_layer=model_config.n_layer))
 

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay, betas=(training_config.beta1,training_config.beta2), foreach=False
    )
    # optimizer = FusedAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2),adam_w_mode=True)
    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "optimizer": optimizer,  "iter_num": 0, "step_count": 0}

    if training_config.resume is True:
        training_config.resume = sorted(training_config.out_dir.glob("*.pth"))[-1]
    if training_config.resume :
        fabric.print(f"Resuming training from {training_config.resume}")
        fabric.load(training_config.resume, state)

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader, monitor, training_config)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(fabric, state, train_dataloader, val_dataloader, monitor, training_config):
    model = state["model"]
    optimizer = state["optimizer"]

    if val_dataloader is not None:
        validate(fabric, model, val_dataloader, training_config)  # sanity check

    with torch.device("meta"):
        meta_model = GPT(model.config)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * training_config.micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (training_config.micro_batch_size, model.config.block_size))
        # measured_flos run in meta. Will trigger fusedRMSNorm error
        #measured_flops = measure_flops(meta_model, x)
        #fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    total_lengths = 0
    total_t0 = time.perf_counter()

    
    
    initial_iter = state["iter_num"]
    curr_iter = 0
            
    loss_func = FusedCrossEntropyLoss()
    for  train_data in train_dataloader:
        # resume loader state. This is not elegant but it works. Should rewrite it in the future.
        if training_config.resume:
            if curr_iter < initial_iter:
                curr_iter += 1
                continue
            else:
                training_config.resume = False
                curr_iter = -1
                fabric.barrier()
                fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
        if state["iter_num"] >= training_config.max_iters:
            break
        
        # determine and set the learning rate for this iteration
        lr = get_cosine_lr(state["iter_num"], training_config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0 : model.config.block_size].contiguous()
        targets = train_data[:, 1 : model.config.block_size + 1].contiguous()
        is_accumulating = (state["iter_num"] + 1) % training_config.gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = loss_func(logits, targets)
            # loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            fabric.backward(loss / training_config.gradient_accumulation_steps)
        with torch.no_grad():
            gathered_loss = fabric.all_reduce(loss)
        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=training_config.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1
        state["iter_num"] += 1
        # input_id: B L 
        total_lengths += input_ids.size(1)
        t1 = time.perf_counter()
        fabric.print(
                f"iter {state['iter_num']} step {state['step_count']}: loss {gathered_loss.item():.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
                f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (training_config.max_iters - state['iter_num']) / 3600:.2f} hours. " 
                # print days as well
                f" or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (training_config.max_iters - state['iter_num']) / 3600 / 24:.2f} days. "
            )
 
        monitor.on_train_batch_end(
            state["iter_num"] * training_config.micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            state["step_count"],
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
            train_loss = gathered_loss.item(),
            lr = lr
        )

            
        if val_dataloader is not None and not is_accumulating and state["step_count"] in training_config.eval_step_list:
            
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader, training_config)
            t1 = time.perf_counter() - t0
            monitor.eval_end(t1)
            fabric.print(f"step {state['iter_num']}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.log_dict({"metric/val_loss": val_loss.item(), "total_tokens": model.config.block_size * (state["iter_num"] + 1) * training_config.micro_batch_size * fabric.world_size}, state["step_count"])
            fabric.log_dict({"metric/val_ppl": math.exp(val_loss.item()), "total_tokens": model.config.block_size * (state["iter_num"] + 1) * training_config.micro_batch_size * fabric.world_size}, state["step_count"])
            fabric.barrier()
        if not is_accumulating and state["step_count"] in training_config.save_step_list:
            checkpoint_path = training_config.out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)

        
@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader, training_config) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(training_config.eval_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= training_config.eval_iters:
            break
        input_ids = val_data[:, 0 : model.config.block_size].contiguous()
        targets = val_data[:, 1 : model.config.block_size + 1].contiguous()
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)

        # loss_func = FusedCrossEntropyLoss()
        # loss = loss_func(logits, targets)
        losses[k] = loss.item()
        
    out = losses.mean()
    fabric.all_reduce(out)
    return out


def create_dataloader(
    training_config,  fabric, shuffle: bool = True, split="train"
) -> DataLoader:
    datasets = []
    data_config = training_config.train_data_config if split == "train" else training_config.val_data_config
    data_dir = training_config.train_data_dir if split == "train" else training_config.val_data_dir 
    for prefix, _ in data_config:
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))
        random.seed(training_config.seed)
        random.shuffle(filenames)

        dataset = PackedDataset(
            filenames,
            # n_chunks control the buffer size. 
            # Note that the buffer size also impacts the random shuffle
            # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
            n_chunks=8,
            block_size=training_config.model_config.block_size+1,
            shuffle=shuffle,
            seed=training_config.seed+fabric.global_rank,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=training_config.seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=training_config.micro_batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    fabric, training_config
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    train_dataloader = create_dataloader(
        training_config=training_config,
        fabric=fabric,
        shuffle=True,
        split="train"
    )
    val_dataloader = (
        create_dataloader(
            training_config=training_config,
            fabric=fabric,
            shuffle=False,
            split="validation"
        )
        if training_config.val_data_dir
        else None
    )
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with warmup)
def get_cosine_lr(it, training_config):
    # 1) linear warmup for warmup_iters steps
    if it < training_config.warmup_iters:
        return training_config.learning_rate * it / training_config.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > training_config.lr_decay_iters:
        return training_config.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - training_config.warmup_iters) / (training_config.lr_decay_iters - training_config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return training_config.min_lr + coeff * (training_config.learning_rate - training_config.min_lr)

def get_eval_step(x):
    """Generate a list of geometric progression between 125 and X with a common ratio of 2."""
    if x <= 0:
        return "X must be greater than 0"

    gp_list = []
    current_value = 125

    while current_value <= x:
        gp_list.append(current_value)
        current_value *= 2

    return gp_list

if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
