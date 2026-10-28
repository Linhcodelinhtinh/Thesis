"""Reproducible domain/task/episode-balanced replay, independent of memory text."""
import hashlib
import math
import numpy as np
from torch.utils.data import Sampler


class BalancedReplaySampler(Sampler):
    def __init__(self, dataset, optimizer_batch_size, base_fraction=.5, seed=42):
        if optimizer_batch_size < 2 or not 0 < base_fraction < 1:
            raise ValueError("Mixed replay needs batch >=2 and both domains")
        self.batch_size = optimizer_batch_size
        self.base_count = round(optimizer_batch_size * base_fraction)
        if not 0 < self.base_count < optimizer_batch_size:
            raise ValueError("Requested mixture rounds to an empty domain")
        self.actual_base_fraction = self.base_count / optimizer_batch_size
        self.seed, self.epoch = seed, 0
        self.size = math.ceil(len(dataset) / optimizer_batch_size) * optimizer_batch_size
        self.groups = {}
        offset = 0
        for episode in dataset.episodes:
            self.groups.setdefault(episode["domain"], {}).setdefault(episode["task"], []).append((offset, episode["length"]))
            offset += episode["length"]
        if set(self.groups) != {"base", "mem"}:
            raise ValueError("Mixed sampler requires base and Mem training episodes")
        self.tasks = {domain: sorted(tasks) for domain, tasks in self.groups.items()}

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.size

    def indices(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        schedule = []
        task_cycles = {d: [] for d in self.tasks}
        for _ in range(self.size // self.batch_size):
            domains = ["base"] * self.base_count + ["mem"] * (self.batch_size - self.base_count)
            rng.shuffle(domains)
            for domain in domains:
                if not task_cycles[domain]:
                    task_cycles[domain] = list(rng.permutation(self.tasks[domain]))
                task = task_cycles[domain].pop()
                choices = self.groups[domain][task]
                start, length = choices[int(rng.integers(len(choices)))]
                schedule.append(start + int(rng.integers(length)))
        return schedule

    def __iter__(self):
        return iter(self.indices())

    def schedule_sha256(self):
        return hashlib.sha256(np.asarray(self.indices(), dtype="<i8").tobytes()).hexdigest()
