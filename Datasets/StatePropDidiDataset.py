from JimmyTorch.Datasets import *
from .DidiDataset import DidiDataset


class StatePropDidiDataset():

    data_keys = ["traj", "road", "%time", "traj_guess", "query_mask", "traj_len", "road_len",
                    "erase_rate", "query_size", "observe_size", "driver_id", "start_weekday", "start_seconds",
                    "duration", "total_distance", "avg_distance", "start_pos", "end_pos"]

    def __init__(self,
                 ddm: Union["DDPM", "DDIM"],
                 state_shapes: List[tuple[int]],
                 t_distribution: Literal['same', 'stride', 'uniform'],
                 dataset_root: str,
                 city_name: str,
                 pad_to_len: int,
                 min_erase_rate: float,
                 max_erase_rate: float,
                 set_name: Literal['train', 'eval', 'test', 'debug'],
                 batch_size: int,
                 drop_last: bool = False,
                 shuffle: bool = False,
                 ):
        self.ddm = ddm
        self.skip_step = ddm.skip_step
        self.set_name = set_name
        self.pad_to_len = pad_to_len
        self.min_erase_rate = min_erase_rate
        self.max_erase_rate = max_erase_rate
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.dataset_root = dataset_root
        self.city_name = city_name
        self.T = ddm.T
        self.state_shapes = state_shapes
        self.t_distribution = t_distribution

        # Try load a batch of data, so we can initialize a cache
        dummy_dataset = DidiDataset(
            dataset_root,
            city_name,
            pad_to_len=pad_to_len,
            min_erase_rate=min_erase_rate,
            max_erase_rate=max_erase_rate,
            set_name=set_name,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle
        )

        self.cache: dict[str, Any] = dummy_dataset[0]

        # The actual dataset has batch_size = 1, so we can load data one by one
        self.dataset = DidiDataset(
            dataset_root,
            city_name,
            pad_to_len=pad_to_len,
            min_erase_rate=min_erase_rate,
            max_erase_rate=max_erase_rate,
            set_name=set_name,
            batch_size=1,
            drop_last=False,
            shuffle=shuffle
        )

        # [T, T-s, T-2s, ..., k], k >= 0
        self.t_schedule = list(range(self.T - 1, 0, -self.skip_step))
        if self.t_schedule[-1] != 0:
            self.t_schedule.append(0)
        # [T, T-s, T-2s, ..., k, k-1, ..., 0] if k > 0
        self.t_schedule = torch.tensor(self.t_schedule, dtype=torch.int32, device=DEVICE)
        self.diff_steps = len(self.t_schedule)

        # Each data sample is used for diff_steps iterations
        # There are len(dataset) samples in the dataset
        # In the last diff_steps iterations, we cannot have full batch, so dropped
        self.total_iters = len(self.dataset) * self.diff_steps - self.diff_steps

        # The current and the next t value for each sample
        # t_i is initialized to 1, which means t = self.t_schedule[1] = T-s
        # And so t_i+1 = self.t_schedule[0] = T
        self.cache["t_i"] = torch.ones(batch_size, dtype=torch.int32, device=DEVICE)
        # The noise added to traj_0_query to get traj_tnext_query
        # The noise for each sample is different
        self.cache["eps_0:t_next"] = [torch.zeros(self.T, self.pad_to_len, 2, device=DEVICE) for _ in range(self.diff_steps)]
        # The trajectory with noise added to query points
        self.cache["noisy_traj"] = torch.zeros(batch_size, self.T + 1, self.pad_to_len, 2, device=DEVICE)
        # the state from t+1 to T
        self.cache["s_t+1:T"] = []
        for shape in state_shapes:
            self.cache["s_t+1:T"].append(torch.zeros(batch_size, *shape, device=DEVICE))

        self.B_range = torch.arange(self.batch_size, device=DEVICE)


    def loadDataToBatch(self, place_at: int):
        data_dict = self.dataset_iter.next()

        noisy_trajs, eps_0_to_t_next = self.addNoise(data_dict["traj"][0], data_dict["query_mask"])
        self.cache["noisy_traj"][place_at] = noisy_trajs
        self.cache["eps_0:t_next"][place_at] = eps_0_to_t_next

        self.cache["t_i"][place_at] = 1
        for i in range(len(self.cache["s_t+1:T"])):
            self.cache["s_t+1:T"][i][place_at] = torch.zeros_like(self.cache["s_t+1:T"][i][place_at])

        for key in self.data_keys:
            self.cache[key][place_at] = data_dict[key][0]
        # point_mean, point_std and driver_count are constant


    def __len__(self):
        return self.total_iters

    
    def __iter__(self):
        self.dataset_iter = iter(self.dataset)
        self.iter_idx = 0
        # Fill the cache with the first batch
        for b in range(self.batch_size):
            self.loadDataToBatch(b)
        # There B t_i values
        # Make them evenly distributed over t_schedule, i.e. over [1, diff_steps)
        if self.t_distribution == 'same':
            self.cache["t_i"] = torch.ones(self.batch_size, device=DEVICE, dtype=torch.int32)
        elif self.t_distribution == 'stride':
            self.cache["t_i"] = torch.linspace(1, self.diff_steps // 3, self.batch_size,
                                               device=DEVICE, dtype=torch.int32)
        else:
            self.cache["t_i"] = torch.linspace(1, self.diff_steps - 1, self.batch_size,
                                               device=DEVICE, dtype=torch.int32)
        return self


    def __next__(self):
        if self.iter_idx >= self.total_iters:
            raise StopIteration

        t = self.t_schedule[self.cache["t_i"]]
        t_next = self.t_schedule[self.cache["t_i"] - 1]

        # Get the current batch
        batch = {
            "t": t,            # (B,)
            "t_next": t_next,   # (B,)
            "traj_0": self.cache["traj"],               # (B, L, 2)
            "traj_t": self.cache["noisy_traj"][self.B_range, t],   # (B, L, 2)
            "traj_t+1": self.cache["noisy_traj"][self.B_range, t_next],  # (B, L, 2)
            "traj_T": self.cache["noisy_traj"][:, -1],  # (B, L, 2)
            "eps_0:t": [self.cache["eps_0:t_next"][i][t[i]].clone() for i in range(self.batch_size)],  # B * (L_query, 2)
            "eps_0:t+1": [self.cache["eps_0:t_next"][i][t_next[i]].clone() for i in range(self.batch_size)],  # B * (L_query, 2)
            "s_t+1": [each.clone() for each in self.cache["s_t+1:T"]],  # n_features * (B, *feature_dims)
        }

        for key in self.data_keys[1:]:
            batch[key] = self.cache[key].clone()

        # Update time steps, if any of them reach the end
        # Reload the data inplace of the finished ones
        self.cache["t_i"] += 1
        min_tau_next_ids = torch.argwhere(self.cache["t_i"] == self.diff_steps)
        for min_tau_next_id in min_tau_next_ids:
            self.loadDataToBatch(min_tau_next_id)

        # Update the index
        self.iter_idx += 1

        return batch


    def updateState(self, s_t_to_T: List[torch.Tensor]):
        for i in range(len(s_t_to_T)):
            self.cache["s_t+1:T"][i] = s_t_to_T[i]


    # @torch.compile
    def addNoise(self, traj, query_mask):
        # traj: (L, 2)
        # query_mask: (L)
        mask = query_mask > 0.5
        query_subtraj = traj[mask]  # (M, 2)

        step_noises = torch.randn(self.T, *query_subtraj.shape).to(DEVICE)
        comb_noises = step_noises.clone()

        noisy_trajs = traj.unsqueeze(0).repeat(self.T + 1, 1, 1)

        query_subtraj_t = query_subtraj
        for t in range(0, self.T):
            comb_noises[t] = self.ddm.combineNoise(comb_noises[t - 1], step_noises[t - 1], t - 1)
            query_subtraj_t = self.ddm.diffusionForwardStep(query_subtraj_t, t - 1, step_noises[t - 1])
            noisy_trajs[t+1][mask] = query_subtraj_t

        return noisy_trajs, comb_noises

