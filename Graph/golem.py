
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import pytorch_lightning as pl
import os
import wandb
from einops import rearrange

from Caulimate.Utils.Tools import check_tensor, check_array
from Caulimate.Utils.Lego import CustomMLP, PartiallyPeriodicMLP
from Caulimate.Utils.Visualization import save_DAG

from torch.nn.utils import spectral_norm

class TimeDomainEncoder(nn.Module):
    def __init__(self, m_embed_dim, hid_dim, t_period, periodic_ratio=0.2):
        super(TimeDomainEncoder, self).__init__()
        self.m_embedding = nn.Embedding(12, m_embed_dim)
        self.m_encoder = CustomMLP(m_embed_dim, hid_dim, 1, 3)
        
        self.t_encoder = PartiallyPeriodicMLP(1, hid_dim, 1, t_period, periodic_ratio)
        
        self.fc = CustomMLP(2*hid_dim, hid_dim, 1, 3)
        # init.uniform_(self.fc1.bias, -5000, 5000)

    def forward(self, t):
        m = (t % 12).to(torch.int64).squeeze(dim=1) # 12 months
        m_embed = self.m_embedding(m)
        t_embed = t # no embedding for smoothness
        xm = 1.0 * self.m_encoder(m_embed)
        xt = 1.0 * self.t_encoder(t_embed)
        
        return xt + xm 

class golem(pl.LightningModule):
    pass

class time_vary_golem(pl.LightningModule):
    def __init__(self, 
                save_dir,
                d_X, 
                m_embed_dim, 
                encoder_hid_dim, 
                t_period,
                eu_distance, 
                loss,
                sparse_tol=0.1,
                lr=1e-4, 
                equal_variances=True, 
                seed=1, 
                B_init=None):
        super().__init__()
        self.save_hyperparameters()
        self.save_dir = save_dir
        self.d_X = d_X
        self.seed = seed
        self.lr = lr
        self.equal_variances = equal_variances
        self.loss = loss
        self.B_init = B_init
        self.m_embed_dim = m_embed_dim
        self.encoder_hid_dim = encoder_hid_dim
        self.eu_distance = eu_distance
        self.tol = sparse_tol
        
        self.gradient = []
        self.Bs = np.empty((0, self.d_X, self.d_X))
        # self.encoders = nn.ModuleList([TimeDomainEncoder(m_embed_dim, encoder_hid_dim, cos_len, periodic_ratio=0.1) 
        #                                       for _ in range(self.d_X ** 2 - self.d_X - (self.d_X - self.distance) * (self.d_X - self.distance - 1))])
        self.encoders = nn.ModuleList([TimeDomainEncoder(m_embed_dim, encoder_hid_dim, t_period, periodic_ratio=0.1) for _ in range(self.d_X ** 2)])
        if B_init is not None:
            self.B = nn.Parameter(rearrange(check_tensor(B_init), 'i j -> 1 i j'))
        else:
            self.B = nn.Parameter(torch.randn(1, self.d_X, self.d_X))

    def prepare_data(self) -> None:
        return super().prepare_data()
    
    def apply_spectral_norm(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                spectral_norm(module)
        
    def generate_B(self, T):
        #T_embed = (1 / self.alpha) * torch.cos(self.beta * T + self.bias)
        B = []
        for encoder in self.encoders:
            B_i = encoder(T)
            B_i_sparse = B_i.masked_fill_(torch.abs(B_i) < self.tol, 0)
            B.append(B_i_sparse)
        B = torch.cat(B, dim=1)
        #B = self.reshape_B(B)
        B = B.reshape(-1, self.d_X, self.d_X)# + self.B
        return B
        
    def _preprocess(self, B):
        B = B.clone()
        B_shape = B.shape
        if len(B_shape) == 3:  # Check if B is a batch of matrices
            for i in range(B_shape[0]):  # Iterate over each matrix in the batch
                B[i].fill_diagonal_(0)
        else:
            print("Input tensor is not a batch of matrices.")
            B.data.fill_diagonal_(0)
        return B
        
    def forward(self, T, B_label=None):
        B = self.generate_B(T)
        return B
        
    def training_step(self, batch, batch_idx):
        X, T, Bs, coords = batch
        B_est = self.forward(T)
        losses = self.compute_loss(X, T, B_est)
        for name in losses.keys():
            self.log(name, losses[name])
        
        return losses['total_loss']
    
    def validation_step(self, batch, batch_idx):
        if self.current_epoch % 100 != 0:
            return
        X, T, Bs, coords = batch
        Bs_pred = check_array(self(T).permute(0, 2, 1))
        # if args.dataset != 'synthetic':
        #     Bs_gt = Bs_pred
        #     for i in range(args.num):
        #         Bs_gt[i] = postprocess(Bs_gt[i])
        save_DAG(X.shape[0], self.save_dir, self.current_epoch, Bs_pred, Bs, graph_thres=0.3, add_value=False)
        wandb.save(os.path.join(wandb.run.dir, f'epoch_{self.current_epoch}.h5'))
    
    def reshape_B(self, B):
        B_zeros = check_tensor(torch.zeros(B.shape[0], self.d_X, self.d_X))
        idx = 0
        for i in range(self.d_X ** 2):
            row = i // self.d_X
            col = i % self.d_X
            if -self.distance <= col - row <= self.distance and row != col:
                B_zeros[:, row, col] = B[:, idx]
                idx += 1
            else:
                continue
        return B_zeros

    def compute_loss(self, X, T, B, B_label=None):
        if B_label is not None:
            total_loss = torch.nn.functional.mse_loss(B, B_label)
            losses = {'total_loss': total_loss}
            return losses
        else:
            batch_size = X.shape[0]
            losses = {}
            total_loss = 0
            X = X - X.mean(axis=0, keepdim=True)
            likelihood = torch.sum(self._compute_likelihood(X, B)) / batch_size
            
            for l in self.loss.keys():
                if l == 'L1':
                    #  + torch.sum(self._compute_L1_group_penalty(B))
                    losses[l] = self.loss[l] * (torch.sum(self._compute_L1_penalty(B))) / batch_size
                    total_loss += losses[l]
                elif l == 'dag':
                    losses[l] = self.loss[l] * torch.sum(self._compute_h(B)) / batch_size
                    total_loss += losses[l]
                elif l == 'grad':
                    losses[l] = self.loss[l] * torch.sum(self._compute_gradient_penalty(B, T)) / batch_size
                    total_loss += losses[l]
                elif l == 'flat':
                    losses[l] = self.loss[l] * torch.sum(torch.pow(B[:, 1:] - B[:, :-1], 2)) / batch_size
                    total_loss += losses[l]
            
            losses['likelihood'] = likelihood
            losses['total_loss'] = total_loss + likelihood
            #self.gradient.append(self._compute_gradient_penalty(losses['total_loss']).cpu().detach().item())

            return losses
        
    def _compute_likelihood(self, X, B):
        X = X.unsqueeze(2)
        if self.equal_variances:
            return 0.5 * self.d_X * torch.log(
                torch.square(
                    torch.linalg.norm(X - B @ X)
                )
            ) - torch.linalg.slogdet(check_tensor(torch.eye(self.d_X)) - B)[1]
        else:
            return 0.5 * torch.sum(
                torch.log(
                    torch.sum(
                        torch.square(X - B @ X), dim=0
                    )
                )
            ) - torch.linalg.slogdet(check_tensor(torch.eye(self.d_X)) - B)[1]

    def _compute_L1_penalty(self, B):
        return torch.norm(B, p=1, dim=(-2, -1)) 
   
    def _compute_L1_group_penalty(self, B):
        return torch.norm(B, p=2, dim=(0))

    def _compute_h(self, B):
        matrix_exp = torch.exp(torch.abs(torch.matmul(B, B)))
        traces = torch.sum(torch.diagonal(matrix_exp, dim1=-2, dim2=-1), dim=-1) - B.shape[1]
        return traces

    def _compute_smooth_penalty(self,B_t):
        B = B_t.clone().data
        batch_size = B.shape[0]
        for i in range(batch_size):
            b_fft = torch.fft.fft2(B[i])
            b_fftshift = torch.fft.fftshift(b_fft)
            center_idx = b_fftshift.shape[0] // 2
            b_fftshift[center_idx, center_idx] = 0.0
            b_ifft = torch.fft.ifft2(torch.fft.ifftshift(b_fftshift))
            B[i] = b_ifft
            
        return torch.norm(B, p=1, dim=(-2, -1))
    
    def _compute_gradient_penalty(self, loss):
        gradients = torch.autograd.grad(outputs=loss, inputs=self.linear1.parameters(), retain_graph=True)
        gradient_norm1 = torch.sqrt(sum((grad**2).sum() for grad in gradients))
        
        gradients = torch.autograd.grad(outputs=loss, inputs=self.linear1.parameters(), retain_graph=True)
        gradient_norm2 = torch.sqrt(sum((grad**2).sum() for grad in gradients))
        
        return gradient_norm1 + gradient_norm2
    
    def configure_optimizers(self):
        opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0001)
        return [opt_v], []
    