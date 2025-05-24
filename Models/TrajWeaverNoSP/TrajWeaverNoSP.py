from .DenoingUNet import DenoisingUNet
from .Embedder import MixedCondEmbedder
from JimmyTorch.Models import *
import matplotlib.pyplot as plt
from JimmyTorch.Datasets import plotTraj, geometricDistance

class TrajWeaverNoSP(JimmyModel):
    def __init__(self,
                 ddm: Any,
                 d_in: int,
                 d_out: int,
                 d_list: list[int],
                 d_embed: int,
                 n_heads: int,
                 **JM_kwargs):

        super().__init__(**JM_kwargs)
        self.ddm = ddm

        # The conditional feature embedder module
        self.embedder = MixedCondEmbedder(32, d_embed)
        self.embedder.addVector("start_pos", 2, 16)
        self.embedder.addVector("end_pos", 2, 16)
        self.embedder.addVector("avg_distance", 1, 16)
        self.embedder.addVector("total_distance", 1, 16)
        self.embedder.addVector("duration", 1, 16)
        self.embedder.addCategorical("start_weekday", 7, 16)
        self.embedder.addCategorical("start_minute", 24 * 60, 64)
        self.embedder.addCategorical("traj_len", 513, 64)

        # The denoising UNet module
        self.denoising_unet = DenoisingUNet(d_in=d_in, d_out=d_out, d_list=d_list, d_cond=d_embed, n_heads=n_heads)

        self.train_loss_names = ["Train_MSE"]
        self.eval_loss_names = ["Eval_MSE", "Geo_Dist"]
        self.mse_func = MaskedLoss(nn.MSELoss())
        self.geo_dist_func = MaskedLoss(geometricDistance)


    def forward(self,
                noisy_traj: Tensor,
                ddm_t: Tensor,
                **kw_cond) -> Tuple[Tensor, Tensor]:

        kw_cond["ddm_t"] = ddm_t
        cond_embed = self.embedder(**kw_cond)
        # Pass the noisy input and state features through the UNet
        noise_pred = self.denoising_unet(torch.cat([noisy_traj, kw_cond["road"], kw_cond["%time"],
                                                    kw_cond["traj_guess"], kw_cond["point_type"]], dim=2), cond_embed)
        # Return the predicted noise
        return noise_pred

    def trainStep(self, data_dict) -> (dict[str, Any], dict[str, Any]):
        device = data_dict['traj'].device
        B = data_dict['traj'].shape[0]

        # Generate random time steps, and also random Gaussian noise
        t = torch.randint(0, self.ddm.T, (B,), device=device).long()
        noise = torch.randn_like(data_dict['traj'])

        # query_mask (B, L, 2), 1 for elements to be erased (recovered), 0 for elements that are observed
        query_mask = (data_dict['point_type'] > 0).to(torch.float32)
        noisy_traj = self.ddm.diffuse(data_dict['traj'], t, noise)
        # Only query_mask=1 part of the trajectory will be replaced by the noisy trajectory
        partial_noisy_traj = noisy_traj * query_mask + data_dict['traj'] * (1 - query_mask)

        if self.mixed_precision:
            # Automatic Mixed Precision (AMP) forward pass and loss calculation
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred_noise = self(partial_noisy_traj, t, **data_dict)
                loss = self.mse_func(pred_noise, noise, query_mask)

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
            pred_noise = self(partial_noisy_traj, t, **data_dict)
            loss = self.mse_func(pred_noise, noise, query_mask)
            loss.backward()

            # Standard gradient clipping (if specified)
            if self.clip_grad > 0:
                nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad)

            # Optimizer step
            self.optimizer.step()

        # Zero the gradients
        self.optimizer.zero_grad()

        return {"Train_MSE": loss.item()}, {"output": pred_noise.detach()}

    def evalStep(self, data_dict) -> (dict[str, Any], dict[str, Any]):

        query_mask = (data_dict['point_type'] > 0).to(torch.float32)
        partial_traj_T = torch.randn_like(data_dict['traj']) * query_mask + data_dict['traj'] * (1 - query_mask)

        def predFunc(x_t, t):
            # Important! the denoising is applied to whole trajectory, including the observed part
            # So here we need to recover the observed part
            x_t = x_t * query_mask + data_dict['traj'] * (1 - query_mask)
            pred_noise = self(x_t, t, **data_dict)
            return None, pred_noise

        with torch.no_grad():
            partial_traj_0 = self.ddm.denoise(partial_traj_T, predFunc)
            loss = self.mse_func(partial_traj_0, data_dict['traj'], query_mask).item()
            mean = data_dict["point_mean"].view(1, 1, 2)
            std = data_dict["point_std"].view(1, 1, 2)
            partial_traj_0_raw = partial_traj_0 * std + mean
            traj_raw = data_dict['traj'] * std + mean
            geo_dist = self.geo_dist_func(partial_traj_0_raw, traj_raw, query_mask).item()

        traj_recon = (partial_traj_0 * query_mask + data_dict['traj'] * (1 - query_mask))[:20]
        traj_gt = data_dict["traj"][:20]
        traj_lens = data_dict["traj_len"][:20]

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

        return {"Eval_MSE": loss, "Geo_Dist": geo_dist}, {"output": traj_recon.detach(), "fig": fig}



