from JimmyTorch.Datasets import *
import torch
from typing import *
from rich.status import Status
import os


"""
The dataset definition during pre-processing is as follows:

traj length in [32, 512]

processed_dataset = {
        "traj": [],                 # trajectory: (L_traj, 2) fp32
        "traj_len": [],             # number of points in each trajectory int32
        "match_road": [],           # matched road: (L_road, 2) fp32
        "road_len": [],             # number of points in each matched road int32
        "%time": [],                # % of each point in the whole trip (L_traj, 1) fp32
        "driver_id": [],            # driver ID int32
        "start_weekday": [],        # start weekday int32
        "start_second": [],         # start second within a day int32
        "duration": [],             # duration in seconds int32
        "point_mean": None,         # mean of points in trajectory (1, 2) fp32
        "point_std": None,          # std of points in trajectory (1, 2) fp32
        "driver_count": 0,          # number of drivers int32
    }
"""

class DidiDataset(JimmyDataset):
    def __init__(self,
                 dataset_root: str,
                 city_name: Literal['Xian', 'Chengdu'],
                 set_name: Literal['train', 'eval', 'test', 'debug'],
                 pad_to_len: int,
                 min_erase_rate: float,
                 max_erase_rate: float,
                 batch_size: int,
                 drop_last: bool = False,
                 shuffle: bool = False,
                 ):
        super().__init__(batch_size, drop_last, shuffle)

        self.set_name = set_name
        self.pad_to_len = pad_to_len
        self.min_erase_rate = min_erase_rate
        self.max_erase_rate = max_erase_rate

        load_path = os.path.join(dataset_root, f"{city_name}_{set_name}.pth")

        with Status(f'Loading {load_path} from disk...'):
            dataset = torch.load(load_path)

            # --- Sequential Features ---
            self.trajs = dataset["traj"]
            self.match_roads = dataset["match_road"]  # B * (L_road[b], 2)
            self.percent_times = dataset["%time"]  # B * (L_traj[b], )

            # --- Integer Features ---
            self.traj_lens = dataset["traj_len"]
            self.road_lens = dataset["road_len"]
            self.driver_ids = dataset["driver_id"]
            self.start_weekdays = dataset["start_weekday"]
            self.start_seconds = dataset["start_second"]
            self.durations = dataset["duration"]

            # --- Constant Features ---
            self.point_mean = dataset["point_mean"].to(DEVICE)  # (1, 2)
            self.point_std = dataset["point_std"].to(DEVICE)  # (1, 2)
            self.driver_count = dataset["driver_count"]  # int

            self.n_samples = len(self.trajs)

            # Unify the length of sequences
            for b in range(len(self.trajs)):
                # Trajectories are 0-padded
                self.trajs[b] = cropPadTraj(self.trajs[b], self.pad_to_len, 0.0)
                # times are 1-padded, because %time for each traj is 0.0 to 1.0
                self.percent_times[b] = cropPadSequence(self.percent_times[b].unsqueeze(-1), self.pad_to_len, 1.0)
                # matched roads are interpolated to the same length
                # because map-matched traj has different lengths compared to the original traj
                self.match_roads[b] = interpTraj(self.match_roads[b], self.pad_to_len, mode="linear")

            self.trajs = torch.stack(self.trajs, dim=0)                     # (B, L_max, 2)
            self.traj_roads = torch.stack(self.match_roads, dim=0)           # (B, L_max, 2)
            self.percent_times = torch.stack(self.percent_times, dim=0)     # (B, L_max, 1)


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
            "%time": percent_times,  # (B, L)
            "traj_guess": trajs_guess,  # (B, L)
            "query_mask": query_mask,   # (B, L)
            "traj_len": traj_lens,      # (B, )
            "road_len": self.road_lens[indices].to(DEVICE),     # (B, )
            "erase_rate": erase_rates,              # (B, )
            "query_size": n_erased,                 # (B, )
            "observe_size": traj_lens - n_erased,   # (B, )
            "driver_id": self.driver_ids[indices].to(DEVICE),           # (B, )
            "start_weekday": self.start_weekdays[indices].to(DEVICE),   # (B, )
            "start_second": self.start_seconds[indices].to(DEVICE),    # (B, )
            "duration": self.durations[indices].to(DEVICE),             # (B, )
            "point_mean": self.point_mean,      # (1, 2)
            "point_std": self.point_std,        # (1, 2)
            "driver_count": self.driver_count,  # python int
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
            load_path="/home/jimmy/Data/Didi/Xian_processed.pt",
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