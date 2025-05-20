from .DenoingUNet import DenoisingUNet
from .Embedder import MixedCondEmbedder
from .StatePropagator import MultiLevelStagePropagator
from JimmyTorch.Models import *
import matplotlib.pyplot as plt
from JimmyTorch.Datasets import plotTraj, geometricDistance

class TrajWeaver(JimmyModel):
    def __init__(self,
                 ddm: Any,
                 d_in: int,
                 d_out: int,
                 d_state: int,
                 d_list: list[int],
                 d_embed: int,
                 l_traj: int,
                 n_heads: int,
                 **JM_kwargs):

        super().__init__(**JM_kwargs)
        self.ddm = ddm

        # The conditional feature embedder module
        self.embedder = MixedCondEmbedder(d_embed)
        self.embedder.addVector("start_pos", 2, 16)
        self.embedder.addVector("end_pos", 2, 16)
        self.embedder.addVector("avg_distance", 1, 16)
        self.embedder.addCategorical("start_weekday", 7, 16)
        self.embedder.addCategorical("start_minute", 24 * 60, 64)
        self.embedder.addCategorical("traj_len", 513, 64)

        # The denoising UNet module
        self.denoising_unet = DenoisingUNet(d_in=d_in, d_out=d_out, d_state=d_state, d_list=d_list)

        # Initialize the state features
        state_shapes = self.getStateShapes(l_traj, d_state, len(d_list) - 1)
        self.initial_state = nn.ParameterList([
            nn.Parameter(torch.randn(shape)) for shape in state_shapes
        ])

        # The state propagator module
        self.state_propagator = MultiLevelStagePropagator(d_state, d_embed, n_heads)

        self.train_loss_names = ["MSE(0:t+1)", "MSE(0:t)", "Train_MSE"]
        self.eval_loss_names = ["Eval_MSE", "Geo_Dist"]
        self.mse_func = MaskedLoss(nn.MSELoss())
        self.geo_dist_func = MaskedLoss(geometricDistance)


    def setInitialState(self, ddm_t: Tensor, state_features: List[Tensor]) -> List[Tensor]:
        indices = torch.where(ddm_t == self.ddm.T - 1)[0]
        for b in indices:
            # If a data sample has ddm_t = ddm.T - 1, which means it is the last step of the diffusion process,
            # that is, the first denoising step, then its state features are all initialized to 0
            # Here we need to set the state features to the initial state
            for i in range(len(state_features)):
                state_features[i][b] = state_features[i][b] + self.initial_state[i]
        return state_features


    def forward(self,
                noisy_input: Tensor,
                ddm_t: Tensor,
                state_features: List[Tensor],
                **kw_cond: Any) -> Tuple[Tensor, Tensor]:
        """
        :param noisy_input: The noisy input of shape (B, l_traj, d_in)
        :param ddm_t: The time step of the diffusion model (B, )
        :param state_features: The state features, each state feature is a tensor of shape (B, l_feature, d_state)
        :param kw_cond: The conditional features, each feature has been registered in the embedder
        :return: The predicted noise of shape (B, l_traj, d_out), and the new state features
        """
        state_features = self.setInitialState(ddm_t, state_features)
        # Embed the conditional features
        cond_embed = self.embedder(**kw_cond)
        # Propagate the state features
        state_features = self.state_propagator(state_features, cond_embed)
        # Pass the noisy input and state features through the UNet
        noise_pred, new_state_features = self.denoising_unet(noisy_input, state_features)
        # Return the predicted noise
        return noise_pred, new_state_features


    def __prepareInputs(self, data_dict: dict[str, Any]) -> dict[str, Any]:
        concat_keys = ["road", "%time", "traj_guess", "point_type"]
        concat_features = torch.cat([data_dict[k] for k in concat_keys], dim=2)  # (B, L, 6)
        data_dict["inputs_tnext"] = torch.cat([data_dict["traj_tnext"], concat_features], dim=2)  # (B, L, 8)
        data_dict["inputs_t"] = torch.cat([data_dict["traj_t"], concat_features], dim=2)  # (B, L, 8)
        return data_dict

    @staticmethod
    def getStateShapes(traj_len: int, d_state, n_stages: int) -> list[tuple[int, int]]:
        L = traj_len
        down_state_shapes = [(d_state, L // (2 ** i)) for i in range(n_stages)]
        mid_state_shape = (d_state, L // (2 ** n_stages))
        up_state_shapes = list(reversed([(d_state, L // (2 ** i)) for i in range(1, n_stages + 1)]))
        return down_state_shapes + [mid_state_shape] + up_state_shapes


    def trainStep(self, data_dict) -> (dict[str, Any], dict[str, Any]):
        device = data_dict['traj_0'].device

        data_dict = self.__prepareInputs(data_dict)

        query_mask = (data_dict['point_type'] > 0).to(torch.float32)

        if self.mixed_precision:
            # Automatic Mixed Precision (AMP) forward pass and loss calculation
            with torch.autocast(device_type=device, dtype=torch.float16):
                pred_eps_0_to_tp1, state_features = self.denoising_unet(
                    data_dict['inputs_tnext'],
                    data_dict['s_tnext'],
                )

                pred_eps_0_to_t, _ = self.denoising_unet(
                    data_dict['inputs_t'],
                    state_features,
                )

                loss_0_to_tp1 = self.mse_func(pred_eps_0_to_tp1, data_dict['eps_0:tnext'], query_mask)
                loss_0_to_t = self.mse_func(pred_eps_0_to_t, data_dict['eps_0:t'], query_mask)
                loss = loss_0_to_tp1 + loss_0_to_t

            # Backward pass with AMP
            self.scaler.scale(loss).backward()

            # Gradient clipping with AMP (if specified)
            if self.clip_grad > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad)

            # Optimizer step with AMP
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard forward pass and backward pass
            pred_eps_0_to_tp1, state_features = self.denoising_unet(
                data_dict['inputs_tnext'],
                data_dict['s_tnext'],
            )

            pred_eps_0_to_t, _ = self.denoising_unet(
                data_dict['inputs_t'],
                state_features,
            )

            loss_0_to_tp1 = self.mse_func(pred_eps_0_to_tp1, data_dict['eps_0:tnext'], query_mask)
            loss_0_to_t = self.mse_func(pred_eps_0_to_t, data_dict['eps_0:t'], query_mask)
            loss = loss_0_to_tp1 + loss_0_to_t

            loss.backward()

            # Standard gradient clipping (if specified)
            if self.clip_grad > 0:
                nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad)

            # Optimizer step
            self.optimizer.step()

        # Zero the gradients
        self.optimizer.zero_grad()

        loss_dict = {
            "MSE(0:t+1)": loss_0_to_tp1.item(),
            "MSE(0:t)": loss_0_to_t.item(),
            "Train_MSE": loss.item()
        }

        output_dict = {
            "pred_eps_0_to_tp1": pred_eps_0_to_tp1.detach(),
            "pred_eps_0_to_t": pred_eps_0_to_t.detach(),
        }

        return loss_dict, output_dict


    def evalStep(self, data_dict) -> (dict[str, Any], dict[str, Any]):
        device = data_dict['traj'].device
        B = data_dict['traj'].shape[0]

        query_mask = (data_dict['point_type'] > 0).to(torch.float32)

        concat_keys = ["road", "%time", "traj_guess", "point_type"]
        concat_features = torch.cat([data_dict[k] for k in concat_keys], dim=2)  # (B, L, 6)
        state_features = [f.repeat(B, 1, 1) for f in self.initial_state]

        with torch.no_grad():
            cond_embed = self.embedder(**data_dict)

        def predFunc(traj_t, t):
            nonlocal state_features
            traj_t = traj_t * query_mask + data_dict['traj'] * (1 - query_mask)
            input_t = torch.cat([traj_t, concat_features], dim=2)
            noise_pred, state_features = self.denoising_unet(input_t, self.state_propagator(state_features, cond_embed))
            return None, noise_pred

        with torch.no_grad():
            traj_T = torch.randn_like(data_dict['traj'], device=device)
            partial_traj_0 = self.ddm.denoise(traj_T, predFunc)
            loss = self.mse_func(partial_traj_0, data_dict['traj'], query_mask).item()
            mean = data_dict["point_mean"].view(1, 1, 2)
            std = data_dict["point_std"].view(1, 1, 2)
            partial_traj_0_raw = partial_traj_0 * std + mean
            traj_raw = data_dict['traj'] * std + mean
            geo_dist = self.geo_dist_func(partial_traj_0_raw, traj_raw, query_mask).item()

        traj_recon = (partial_traj_0 * query_mask + data_dict['traj'] * (1 - query_mask))[:20]
        traj_gt = data_dict["traj"][:20]
        traj_lens = data_dict["traj_len"][:20]

        # Draw the reconstructed trajectory
        plt.close("all")
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].set_title("Ground Truth")
        ax[1].set_title("Reconstructed Trajectory")
        plotTraj(ax[0], traj_gt, traj_lens, color="blue", linewidth=1, markersize=1)

        for i in range(min(20, traj_recon.shape[0])):
            traj_query_part = traj_recon[i][data_dict['point_type'][i, :, 0] == 1].view(-1, 2)
            traj_observed_part = traj_recon[i][data_dict['point_type'][i, :, 0] == 0].view(-1, 2)
            plotTraj(ax[1], traj_observed_part, color="black", linewidth=1, markersize=1)
            plotTraj(ax[1], traj_query_part, color="red", linewidth=1, markersize=1)

        return {"Eval_MSE": loss, "Geo_Dist": geo_dist}, {"traj_recon": traj_recon.detach(), "fig": fig}



