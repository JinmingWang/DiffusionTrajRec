from .DenoingUNet import DenoisingUNet
from .Embedder import MixedCondEmbedder
from .StatePropagator import MultiLevelStagePropagator
from JimmyTorch.Models import *

"""
Total Training Params:                                                  23.26 M 
fwd MACs:                                                               1.52 GMACs
fwd FLOPs:                                                              3.04 GFLOPS
fwd+bwd MACs:                                                           4.55 GMACs
fwd+bwd FLOPs:                                                          9.13 GFLOPS
"""


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
                 optimizer_cls=None,
                 optimizer_args=None,
                 mixed_precision: bool = False,
                 clip_grad: float = 0.0):

        super().__init__(
            optimizer_cls=optimizer_cls,
            optimizer_args=optimizer_args,
            mixed_precision=mixed_precision,
            clip_grad=clip_grad
        )
        self.ddm = ddm

        # The conditional feature embedder module
        self.embedder = MixedCondEmbedder(d_embed)
        self.embedder.addVector("start_pos", 2, 16)
        self.embedder.addVector("end_pos", 2, 16)
        self.embedder.addVector("avg_mov_dist", 1, 16)
        self.embedder.addCategorical("weekday", 7, 16)
        self.embedder.addCategorical("start_minute", 24 * 60, 64)
        self.embedder.addCategorical("traj_len", 513, 64)

        # The denoising UNet module
        self.denoising_unet = DenoisingUNet(d_in=d_in, d_out=d_out, d_state=d_state, d_list=d_list)

        # Initialize the state features
        state_shapes = self.denoising_unet.getStateShapes(l_traj)
        self.initial_state = nn.ParameterList([
            nn.Parameter(torch.randn(shape)) for shape in state_shapes
        ])

        # The state propagator module
        self.state_propagator = MultiLevelStagePropagator(d_state, d_embed, n_heads)

        self.train_loss_names = ["Train_MSE(0:t+1)", "Train_MSE(0:t)", "Train_MSE(total)"]
        self.eval_loss_names = ["Eval_MSE_REC"]
        self.mse_func = nn.MSELoss()


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
        concat_keys = ["road", "%time", "traj_guess", "query_mask"]
        concat_features = torch.cat([data_dict[k] for k in concat_keys], dim=2)  # (B, L, 6)
        data_dict["inputs_t+1"] = torch.cat([data_dict["traj_t+1"], concat_features], dim=2)  # (B, L, 8)
        data_dict["inputs_t"] = torch.cat([data_dict["traj_t"], concat_features], dim=2)  # (B, L, 8)
        return data_dict


    def trainStep(self, data_dict) -> (dict[str, Any], dict[str, Any]):
        device = data_dict['traj'].device

        data_dict = self.__prepareInputs(data_dict)

        if self.mixed_precision:
            # Automatic Mixed Precision (AMP) forward pass and loss calculation
            with torch.autocast(device_type=device, dtype=torch.float16):
                pred_eps_0_to_tp1, state_features = self.denoising_unet(
                    data_dict['inputs_t+1'],
                    data_dict['t+1'],
                    data_dict['s_t+1'],
                    **data_dict
                )

                pred_eps_0_to_t, _ = self.denoising_unet(
                    data_dict['inputs_t'],
                    data_dict['t'],
                    state_features,
                    **data_dict
                )

                loss_0_to_tp1 = self.mse_fn(pred_eps_0_to_tp1, data_dict['eps_0:t+1'])
                loss_0_to_t = self.mse_fn(pred_eps_0_to_t, data_dict['eps_0:t'])
                loss = loss_0_to_tp1 + loss_0_to_t

            # Backward pass with AMP
            self.scaler.scale(loss).backward()

            # Gradient clipping with AMP (if specified)
            if self.clip_grad > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

            # Optimizer step with AMP
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard forward pass and backward pass
            pred_eps_0_to_tp1, state_features = self.denoising_unet(
                data_dict['traj_t+1'],
                data_dict['t+1'],
                data_dict['s_t+1'],
                **data_dict
            )

            pred_eps_0_to_t, _ = self.denoising_unet(
                data_dict['traj_t'],
                data_dict['t'],
                state_features,
                **data_dict
            )

            loss_0_to_tp1 = self.mse_fn(pred_eps_0_to_tp1, data_dict['eps_0:t+1'])
            loss_0_to_t = self.mse_fn(pred_eps_0_to_t, data_dict['eps_0:t'])
            loss = loss_0_to_tp1 + loss_0_to_t

            loss.backward()

            # Standard gradient clipping (if specified)
            if self.clip_grad > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

            # Optimizer step
            self.optimizer.step()

        # Zero the gradients
        self.optimizer.zero_grad()

        loss_dict = {
            "Train_MSE(0:t+1)": loss_0_to_tp1.item(),
            "Train_MSE(0:t)": loss_0_to_t.item(),
            "Train_MSE(total)": loss.item()
        }

        output_dict = {
            "pred_eps_0_to_tp1": pred_eps_0_to_tp1.detach(),
            "pred_eps_0_to_t": pred_eps_0_to_t.detach(),
        }

        return loss_dict, output_dict


    def evalStep(self, data_dict) -> (dict[str, Any], dict[str, Any]):
        device = data_dict['traj'].device
        B = data_dict['traj'].shape[0]

        concat_keys = ["road", "%time", "traj_guess", "query_mask"]
        concat_features = torch.cat([data_dict[k] for k in concat_keys], dim=2)  # (B, L, 6)
        state_features = [f.repeat(B, 1, 1) for f in self.initial_state]

        with torch.no_grad():
            cond_embed = self.embedder(**data_dict)

        def predFunc(traj_t, t):
            nonlocal state_features
            input_t = torch.cat([traj_t, concat_features], dim=2)
            noise_pred, state_features = self.denoising_unet(input_t, self.state_propagator(state_features, cond_embed))
            return None, noise_pred

        with torch.no_grad():
            traj_T = torch.randn_like(data_dict['traj'], device=device)
            traj_recon = self.ddm.denoise(traj_T, predFunc)

            mask = data_dict['query_mask'] > 0.5    # (B, L)
            traj_query_rec = traj_recon[mask]   # (?, ), depends on how many elements are masked
            traj_query_gt = data_dict['traj'][mask]   # (?, ), depends on how many elements are masked

            loss = self.mse_func(traj_query_rec, traj_query_gt)
        return {"Eval_MSE_REC": loss.item()}, {"traj_recon": traj_recon.detach()}



