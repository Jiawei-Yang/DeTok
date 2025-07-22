import argparse
import datetime
import json
import logging
import os
import sys
import time
from collections import defaultdict, deque
from typing import Any

import torch
import torch.distributed
import wandb
from rich.logging import RichHandler
from typing_extensions import override

from .distributed import get_global_rank, is_enabled, is_main_process

logger = logging.getLogger("DeTok")


def move_to_device(obj: Any, device: torch.device) -> Any:
    """recursively moves tensors in obj to the specified device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(o, device) for o in obj)
    else:
        return obj


class SmoothedValue:
    """track a series of values and provide access to smoothed values over a window or the global series average."""

    def __init__(self, window_size: int = 20, fmt: str | None = None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value: float, num: int = 1) -> None:
        self.deque.append(value)
        self.count += num
        self.total += value * num

    def synchronize_between_processes(self) -> None:
        """distributed synchronization of the metric. warning: does not synchronize the deque!"""
        if not is_enabled():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self) -> float:
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self) -> float:
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self) -> float:
        return self.total / self.count

    @property
    def max(self) -> float:
        return max(self.deque)

    @property
    def value(self) -> float:
        return self.deque[-1]

    @override
    def __str__(self) -> str:
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter: str = "\t", output_file: str | None = None, prefetch: bool = False):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.output_file = output_file
        self.prefetch = prefetch
        logger.info(f"MetricLogger: output_file={output_file}, prefetch={prefetch}")

    def update(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr: str):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    @override
    def __str__(self) -> str:
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self) -> None:
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name: str, meter: SmoothedValue) -> None:
        self.meters[name] = meter

    def dump_in_output_file(self, iteration: int, iter_time: float, data_time: float) -> None:
        if self.output_file is None or not is_main_process():
            return
        dict_to_dump = dict(
            iteration=iteration,
            iter_time=iter_time,
            data_time=data_time,
        )
        dict_to_dump.update({k: v.median for k, v in self.meters.items()})
        with open(self.output_file, "a") as f:
            f.write(json.dumps(dict_to_dump) + "\n")

    def log_every(
        self,
        iterable,
        print_freq: int,
        header: str | None = None,
        n_iterations: int | None = None,
        start_iteration: int = 0,
    ):
        i = start_iteration
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")

        if n_iterations is None:
            try:
                n_iterations = len(iterable)
            except TypeError:
                # iterable doesn't have len, use a default or require user to provide
                raise ValueError("n_iterations must be provided for iterables without __len__")

        space_fmt = ":" + str(len(str(n_iterations))) + "d"

        log_list = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "elapsed: {elapsed_time_str}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_list += ["max mem: {memory:.0f}"]

        log_msg = self.delimiter.join(log_list)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            if self.prefetch:
                obj = move_to_device(obj, torch.device("cuda"))
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == n_iterations - 1:
                self.dump_in_output_file(iteration=i, iter_time=iter_time.avg, data_time=data_time.avg)
                eta_seconds = iter_time.global_avg * (n_iterations - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                elapsed_time = time.time() - start_time
                elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_time)))

                if torch.cuda.is_available():
                    logger.info(
                        log_msg.format(
                            i,
                            n_iterations,
                            eta=eta_string,
                            elapsed_time_str=elapsed_time_str,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    logger.info(
                        log_msg.format(
                            i,
                            n_iterations,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
            if i >= n_iterations:
                break
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f"{header} Total time: {total_time_str} ({total_time / n_iterations:.6f} s / it)")


class WandbLogger:
    def __init__(
        self,
        config,
        entity: str,
        project: str,
        name: str,
        log_dir: str,
        run_id: str | None = None,
    ):
        self.run = wandb.init(
            config=config,
            entity=entity,
            project=project,
            name=name,
            dir=log_dir,
            resume="allow",
            id=run_id,
        )
        self.run_id = self.run.id
        self.step = 0
        self.run.log_code(".")

    def update(self, metrics, step: int | None = None) -> None:
        log_dict = {
            k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items() if v is not None
        }
        try:
            wandb.log(log_dict, step=step or self.step)
        except Exception as e:
            logger.error(f"wandb logging failed: {e}")
        if step is not None:
            self.step = step

    def finish(self) -> None:
        try:
            wandb.finish()
        except Exception as e:
            logger.error(f"wandb failed to finish: {e}")


def setup_logging(output: str, name: str = "DeTok", rank0_log_only: bool = True) -> None:
    """setup logging."""
    logging.captureWarnings(True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # google glog format: [IWEF]yyyymmdd hh:mm:ss logger filename:line] msg
    fmt_prefix = "%(levelname).1s%(asctime)s %(name)s %(filename)s:%(lineno)s] "
    fmt_message = "%(message)s"
    fmt = fmt_prefix + fmt_message
    datefmt = "%Y%m%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # stdout logging for main worker only
    if is_main_process():
        if sys.stdout.isatty():
            handler = RichHandler(markup=True, show_time=False, show_level=False)
        else:
            handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # file logging
    if output:
        if os.path.splitext(output)[-1] in (".txt", ".log"):
            # if output is a file, use it directly
            filename = output
        else:
            # if output is a directory, use the directory/log.txt
            filename = os.path.join(output, "log.txt")

        # if not rank 0 but rank0_log_only=false, append the rank id
        if not is_main_process() and not rank0_log_only:
            global_rank = get_global_rank()
            if filename.endswith(".txt"):
                filename = filename.replace(".txt", f".rank{global_rank}.txt")
            else:
                filename = f"{filename}.rank{global_rank}"

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        handler = logging.StreamHandler(open(filename, "a"))
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def setup_wandb(args: argparse.Namespace, entity: str, project: str, name: str, log_dir: str) -> WandbLogger:
    """Setup Weights & Biases logging with resume capability."""
    run_id_path = os.path.join(log_dir, "wandb_run_id.txt")
    run_id = None

    # resume from wandb run id if it exists
    if os.path.exists(run_id_path):
        with open(run_id_path, "r") as f:
            run_id = f.readlines()[-1].strip()

    wandb_logger = WandbLogger(
        config=args,
        entity=entity,
        project=project,
        name=name,
        log_dir=log_dir,
        run_id=run_id,
    )

    # if no run id, save the run id to the log directory
    if run_id is None:
        with open(run_id_path, "a") as f:
            f.write(wandb_logger.run.id + "\n")

    return wandb_logger
