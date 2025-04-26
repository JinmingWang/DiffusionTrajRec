from JimmyTorch.Datasets import *
import torch
from typing import *
from rich.status import Status


class DidiDataset(JimmyDataset):
    def __init__(self,
                 load_path: str,
                 pad_to_len: int,
                 min_erase_rate: float,
                 max_erase_rate: float,
                 set_name: Literal['train', 'eval', 'test', 'debug'],
                 batch_size: int,
                 drop_last: bool = False,
                 shuffle: bool = False,
                 ):
        super().__init__(batch_size, drop_last, shuffle)

        self.set_name = set_name
        self.pad_to_len = pad_to_len
        self.min_erase_rate = min_erase_rate
        self.max_erase_rate = max_erase_rate

        match set_name:
            case "train":
                start, end = 0.0, 0.8
            case "eval":
                start, end = 0.8, 0.9
            case "test":
                start, end = 0.9, 1.0
            case "debug":
                start, end = 0.0, 0.05
            case _:
                raise ValueError(f"Unknown set_name: {set_name}")

        with Status(f'Loading {load_path} from disk...'):
            dataset = torch.load(load_path)
            self.trajs = dataset["traj"]
            n_trajs = len(self.trajs)
            start = int(n_trajs * start)
            end = int(n_trajs * end) + 1

            self.trajs = self.trajs[start:end]                             # B * (L_traj[b], 2)
            self.traj_roads = dataset["traj_road"][start:end]              # B * (L_road[b], 2)
            self.percent_times = dataset["percent_times"][start:end]       # B * (L_traj[b], )

            # Pad the trajectories to the same length
            for b in range(len(self.trajs)):
                self.trajs[b] = cropPadTraj(self.trajs[b], self.pad_to_len, 0.0)
                self.traj_roads[b] = cropPadTraj(self.traj_roads[b], self.pad_to_len, 0.0)
                self.percent_times[b] = cropPadSequence(self.percent_times[b].unsqueeze(-1), self.pad_to_len, 0.0)

            self.trajs = torch.stack(self.trajs, dim=0)                     # (B, L_max, 2)
            self.traj_roads = torch.stack(self.traj_roads, dim=0)           # (B, L_max, 2)
            self.percent_times = torch.stack(self.percent_times, dim=0)     # (B, L_max, 1)

            self.traj_lens = dataset["traj_len"][start:end]                # (B, )
            self.road_lens = dataset["road_length"][start:end]           # (B, )
            self.driver_ids = dataset["driver_id"][start:end]              # (B, )
            self.start_weekdays = dataset["start_weekday"][start:end]      # (B, )
            self.start_seconds = dataset["start_time_in_day"][start:end]   # (B, )
            self.durations = dataset["duration"][start:end]                # (B, )

            self.point_mean = dataset["point_mean"].to(DEVICE)             # (1, 2)
            self.point_std = dataset["point_std"].to(DEVICE)               # (1, 2)
            self.driver_count = dataset["driver_count"].to(DEVICE)         # torch.int32

        self.n_samples = len(self.trajs)

    @staticmethod
    def guessTraj(traj, times, query_mask):
        """
        Obtain the guessed trajectory from the original trajectory and the erase mask
        :param traj: (L, 2)
        :param times: (L, )
        :param query_mask:  (L, )
        :return: guessed traj: (L, 2)
        """
        boolean_mask = query_mask > 0.1  # (L,)

        # query_subtraj = traj[boolean_mask]  # (L_erased, 2)
        observ_subtraj = traj[~boolean_mask]  # (L_remain, 2)
        query_times = times[boolean_mask]
        observ_times = times[~boolean_mask]

        ids_right = torch.searchsorted(observ_times, query_times).to(torch.long)  # (L_erased)
        ids_left = ids_right - 1  # (L_erased)

        times_left = observ_times[ids_left]
        times_right = observ_times[ids_right]
        ratio = ((query_times - times_left) / (times_right - times_left)).view(-1, 1)  # (L_erased)

        traj_left = observ_subtraj[ids_left]
        traj_right = observ_subtraj[ids_right]
        query_subtraj_guess = traj_left * (1 - ratio) + traj_right * ratio  # (2, L_erased)

        traj_guess = traj.clone()  # (2, L)
        traj_guess[boolean_mask] = query_subtraj_guess

        nan_mask = torch.isnan(traj_guess)
        traj_guess[nan_mask] = torch.zeros_like(traj_guess[nan_mask])

        return traj_guess



    def __getitem__(self, idx):
        start = (idx - 1) * self.batch_size
        end = min(start + self.batch_size, self.n_samples)
        indices = self._indices[start:end]
        B = len(indices)

        trajs = self.trajs[indices].to(DEVICE)
        percent_times = self.percent_times[indices].to(DEVICE)
        traj_lens = self.traj_lens[indices].to(DEVICE)

        erase_rates = torch.rand(B) * (self.max_erase_rate - self.min_erase_rate) + self.min_erase_rate
        # Compute the number of points to erase for each trajectory
        n_erased = (traj_lens * erase_rates).to(torch.long)
        trajs_guess = torch.zeros_like(trajs)
        query_mask = torch.zeros(B, self.pad_to_len, device=DEVICE)
        for b in range(B):
            erase_indices = torch.randperm(traj_lens[b])[:n_erased[b]]
            query_mask[b, erase_indices] = 1.0      # 1 for erased
            query_mask[b, traj_lens[b]:] = -1.0     # -1 for padding
            query_mask[b, [0, traj_lens[b]]] = 0.0           # never erase the first and last point
            trajs_guess[b] = DidiDataset.guessTraj(trajs[b], percent_times[b], query_mask[b])

        return {
            "traj": trajs,  # (B, L, 2)
            "road": self.traj_roads[indices].to(DEVICE),    # (B, L, 2)
            "percent_time": percent_times,  # (B, L)
            "traj_guess": trajs_guess,  # (B, L)
            "query_mask": query_mask,   # (B, L)
            "traj_len": traj_lens,      # (B, )
            "road_len": self.road_lens[indices].to(DEVICE),     # (B, )
            "erase_rate": erase_rates,              # (B, )
            "query_size": n_erased,                 # (B, )
            "observe_size": traj_lens - n_erased,   # (B, )
            "driver_id": self.driver_ids[indices].to(DEVICE),           # (B, )
            "start_weekday": self.start_weekdays[indices].to(DEVICE),   # (B, )
            "start_seconds": self.start_seconds[indices].to(DEVICE),    # (B, )
            "duration": self.durations[indices].to(DEVICE),             # (B, )
            "point_mean": self.point_mean,      # (1, 2)
            "point_std": self.point_std,        # (1, 2)
            "driver_count": self.driver_count,  # torch.int32
        }


class DidiXianNovDataset(DidiDataset):
    def __init__(self,
                 pad_to_len: int,
                 min_erase_rate: float,
                 max_erase_rate: float,
                 set_name: Literal['train', 'eval', 'test'],
                 batch_size: int,
                 drop_last: bool = False,
                 shuffle: bool = False,
                 ):
        super().__init__(
            load_path="/homt/jimmy/Data/Didi/Xian_processed.pt",
            pad_to_len=pad_to_len,
            min_erase_rate=min_erase_rate,
            max_erase_rate=max_erase_rate,
            set_name=set_name,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
        )


class DidiChengduNovDataset(DidiDataset):
    def __init__(self,
                 pad_to_len: int,
                 min_erase_rate: float,
                 max_erase_rate: float,
                 set_name: Literal['train', 'eval', 'test'],
                 batch_size: int,
                 drop_last: bool = False,
                 shuffle: bool = False,
                 ):
        super().__init__(
            load_path="/home/jimmy/Data/Didi/Chengdu_processed.pt",
            pad_to_len=pad_to_len,
            min_erase_rate=min_erase_rate,
            max_erase_rate=max_erase_rate,
            set_name=set_name,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
        )