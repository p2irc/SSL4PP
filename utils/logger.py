"""Logging utilities.

The classes defined in this script were originally implemented in the torchvision
classification references:
https://github.com/pytorch/vision/blob/main/references/classification/utils.py

"""
import datetime
import time
from collections import defaultdict, deque
from typing import Iterable

import torch
import torch.distributed as dist

import utils.distributed


class SmoothedValue:
    """Maintain a smoothed value over a window or number of iterations.

    Track a series of values and provide access to smoothed values over a
    window or the global series average.


    Args:
        window_size: int
            Number of values to track
        fmt: str
            String format to use when computing the __str__ method.

    """

    def __init__(self, window_size: int = 20, fmt: str = None):
        """Init method."""
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        """Update the series with new value."""
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """Synchronize the stats across processes.

        Warning: does not synchronize the deque!

        """
        if not utils.distributed.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        """The median of the values."""
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """The average of the values."""
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        """The average of all values."""
        return self.total / self.count

    @property
    def max(self):
        """The max of all values."""
        return max(self.deque)

    @property
    def value(self):
        """The most recent value."""
        return self.deque[-1]

    def __str__(self):
        """String representation of the class."""
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    r"""Log training progress.

    Args:
        delimiter: str, default: "\\t"
            delimiter to use between metrics.

    """

    def __init__(self, delimiter: str = "\t"):
        """Init method."""
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """Update the metrics."""
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        """Get attribute."""
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        """A string representation of the class."""
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """Synchronize the stats across processes."""
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """Add a meter to the logger."""
        self.meters[name] = meter

    def log_every(self, iterable: Iterable, print_freq: int, header: str = None):
        """Log every print_freq iterations.

        Args:
            iterable: iterable
                Iterable to be logged
            print_freq: int
                Frequency of logging
            header: str, default: None
                String to prepend at the beginning of the logging

        """
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("{} Total time: {}".format(header, total_time_str))
