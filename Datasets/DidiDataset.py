from JimmyTorch.Datasets import *
import torch
from typing import *
from rich.status import Status
import os
import random


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
        "total_distance": [],       # total distance of the trajectory (1,) fp32
        "avg_distance": [],         # average distance between consecutive points (1,) fp32
        "start_pos": [],            # start position of the trajectory (1, 2) fp32
        "end_pos": []               # end position of the trajectory (1, 2) fp32
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
                 compute_guess: bool = True,
                 ):
        super().__init__(batch_size, drop_last, shuffle)

        self.set_name = set_name
        self.pad_to_len = pad_to_len
        self.min_erase_rate = min_erase_rate
        self.max_erase_rate = max_erase_rate

        self.compute_guess = compute_guess

        load_path = os.path.join(dataset_root, f"{city_name}_{set_name}.pth")

        with Status(f'Loading {load_path} from disk...'):
            dataset = torch.load(load_path)

            # --- Sequential Features ---
            self.trajs = dataset["traj"]
            self.match_roads = dataset["match_road"]  # B * (L_traj[b], 2)
            self.percent_times = dataset["%time"]  # B * (L_traj[b], )

            # --- Integer Features ---
            self.traj_lens = dataset["traj_len"].to(DEVICE)
            self.road_lens = dataset["road_len"].to(DEVICE)
            self.driver_ids = dataset["driver_id"].to(DEVICE)
            self.start_weekdays = dataset["start_weekday"].to(DEVICE)
            self.start_seconds = dataset["start_second"].to(DEVICE)
            self.durations = dataset["duration"].to(DEVICE)

            # --- Float Features ---
            self.total_distance = dataset["total_distance"].to(DEVICE)
            self.avg_distance = dataset["avg_distance"].to(DEVICE)
            self.start_pos = dataset["start_pos"].to(DEVICE)
            self.end_pos = dataset["end_pos"].to(DEVICE)

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

            self.trajs = torch.stack(self.trajs, dim=0).to(DEVICE)                     # (B, L_max, 2)
            self.match_roads = torch.stack(self.match_roads, dim=0).to(DEVICE)           # (B, L_max, 2)
            self.percent_times = torch.stack(self.percent_times, dim=0).to(DEVICE)     # (B, L_max, 1)

    def guessTraj(self, traj, times, query_mask):
        """
        Obtain the guessed trajectory from the original trajectory and the erase mask
        :param traj: (L, 2)
        :param times: (L, )
        :param query_mask:  (L, )
        :return: guessed traj: (L, 2)
        """
        if not self.compute_guess:
            return traj

        boolean_mask = query_mask > 0.1  # (L,)

        # query_subtraj = traj[boolean_mask]  # (L_erased, 2)
        observ_subtraj = traj[~boolean_mask]  # (L_remain, 2)
        query_times = times.flatten()[boolean_mask]
        observ_times = times.flatten()[~boolean_mask]

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


    def __consecutiveErase(self, traj_len, n):
        start = random.randint(0, int(traj_len - n))
        erase_indices = torch.arange(start, start + int(n), device=DEVICE)
        return erase_indices


    def __randomErase(self, traj_len, n):
        return torch.randperm(int(traj_len), device=DEVICE)[:n]


    def __hybridErase(self, traj_len, n, *erase_method: Literal["consecutive", "random"]):
        num_methods = len(erase_method)
        erase_nums = torch.rand(num_methods, device=DEVICE)
        erase_nums = ((erase_nums / erase_nums.sum()) * n).to(torch.long)
        erase_indices = []
        for i, method in enumerate(erase_method):
            if method == "consecutive":
                erase_indices.append(self.__consecutiveErase(traj_len, erase_nums[i]))
            else:
                erase_indices.append(self.__randomErase(traj_len, erase_nums[i]))

        erase_indices = torch.cat(erase_indices)
        return torch.unique(erase_indices)



    def makeBrokenTraj(self, trajs, traj_lens, percent_times):
        """
        This function generates data corresponding to trajectory recovery task.
        The data includes erase rates, number of points to erase, guessed trajectories, and point types.

        ! It uses different erase patterns for each trajectory, to better simulate the real-world scenarios.

        :param trajs: Trajectories to be erased
        :param traj_lens: Lengths of the trajectories
        :param percent_times: Percent times of the trajectories
        :return:
        """
        B = len(trajs)
        erase_rates = torch.rand(B, device=DEVICE) * (self.max_erase_rate - self.min_erase_rate) + self.min_erase_rate
        erase_modes = torch.randint(0, 5, (B,), device=DEVICE)
        # erase_modes in {0, 1, 2, 3}
        # 0: Random select points to erase
        # 1: Select n consecutive points within the trajectory to erase
        # 2: Do #2 twice, with n consecutive points and m consecutive points, n+m=n_erased
        # 3: A mixture of #1 and #0
        # 4: A mixture of #2 and #0

        # Compute the number of points to erase for each trajectory
        n_erased = (traj_lens * erase_rates).to(torch.long)
        trajs_guess = torch.zeros_like(trajs)
        point_type = torch.zeros(B, self.pad_to_len, device=DEVICE)
        for b in range(B):
            match erase_modes[b]:
                case 0: # Random select points to erase
                    erase_indices = self.__randomErase(traj_lens[b], n_erased[b])
                case 1: # Select n consecutive points to erase
                    erase_indices = self.__consecutiveErase(traj_lens[b], n_erased[b])
                case 2: # Do #2 twice
                    erase_indices = self.__hybridErase(traj_lens[b], n_erased[b], "consecutive", "consecutive")
                case 3: # A mixture of #1 and #0
                    erase_indices = self.__hybridErase(traj_lens[b], n_erased[b], "random", "consecutive")
                case 4: # A mixture of #2 and #0
                    erase_indices = self.__hybridErase(traj_lens[b], n_erased[b], "random", "consecutive", "consecutive")
                case _: # Default to random select points to erase
                    erase_indices = self.__randomErase(traj_lens[b], n_erased[b])

            point_type[b, erase_indices] = 1.0  # 1 for erased
            point_type[b, traj_lens[b]:] = -1.0  # -1 for padding
            trajs_guess[b] = self.guessTraj(trajs[b], percent_times[b], point_type[b])

        return erase_rates, n_erased, trajs_guess, point_type



    def __getitem__(self, idx):
        start = (idx - 1) * self.batch_size
        end = min(start + self.batch_size, self.n_samples)
        indices = self._indices[start:end]

        trajs = self.trajs[indices]
        percent_times = self.percent_times[indices]
        traj_lens = self.traj_lens[indices]

        erase_rates, n_erased, trajs_guess, point_type = self.makeBrokenTraj(trajs, traj_lens, percent_times)

        return {
            "traj": trajs,  # (B, L, 2)
            "road": self.match_roads[indices],    # (B, L, 2)
            "%time": percent_times,  # (B, L)
            "traj_guess": trajs_guess,  # (B, L)
            "point_type": point_type,   # (B, L)
            "traj_len": traj_lens,      # (B, )
            "road_len": self.road_lens[indices],     # (B, )
            "erase_rate": erase_rates,              # (B, )
            "query_size": n_erased,                 # (B, )
            "observe_size": traj_lens - n_erased,   # (B, )
            "driver_id": self.driver_ids[indices],           # (B, )
            "start_weekday": self.start_weekdays[indices],   # (B, )
            "start_second": self.start_seconds[indices],    # (B, )
            "duration": self.durations[indices],             # (B, )
            "point_mean": self.point_mean,      # (1, 2)
            "point_std": self.point_std,        # (1, 2)
            "driver_count": self.driver_count,  # python int
            "total_distance": self.total_distance[indices],
            "avg_distance": self.avg_distance[indices],
            "start_pos": self.start_pos[indices],
            "end_pos": self.end_pos[indices]
        }


class DidiXianNovDataset(DidiDataset):
    def __init__(self,
                 dataset_root: str,
                 pad_to_len: int,
                 min_erase_rate: float,
                 max_erase_rate: float,
                 set_name: Literal['train', 'eval', 'test'],
                 batch_size: int,
                 drop_last: bool = False,
                 shuffle: bool = False,
                 compute_guess: bool = True
                 ):
        super().__init__(
            dataset_root=dataset_root,
            city_name="Xian",
            pad_to_len=pad_to_len,
            min_erase_rate=min_erase_rate,
            max_erase_rate=max_erase_rate,
            set_name=set_name,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            compute_guess=compute_guess
        )


class DidiChengduNovDataset(DidiDataset):
    def __init__(self,
                 dataset_root: str,
                 pad_to_len: int,
                 min_erase_rate: float,
                 max_erase_rate: float,
                 set_name: Literal['train', 'eval', 'test'],
                 batch_size: int,
                 drop_last: bool = False,
                 shuffle: bool = False,
                 compute_guess: bool = True
                 ):
        super().__init__(
            dataset_root=dataset_root,
            city_name="Xian",
            pad_to_len=pad_to_len,
            min_erase_rate=min_erase_rate,
            max_erase_rate=max_erase_rate,
            set_name=set_name,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            compute_guess=compute_guess
        )