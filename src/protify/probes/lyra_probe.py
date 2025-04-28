import torch
import math
import torch.nn as nn
from typing import Optional
from einops import rearrange, repeat
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput
from model_components.mlp import intermediate_correction_fn
from .losses import get_loss_fct


class PGC(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        expansion_factor: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size # d
        self.expansion_factor = expansion_factor # E / d
        self.dropout = dropout
        self.in_proj = nn.Linear(input_size, int(hidden_size * expansion_factor * 2))
        self.in_norm = nn.RMSNorm(int(hidden_size * expansion_factor * 2))

        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size)

        self.out_proj = nn.Linear(int(hidden_size * expansion_factor), input_size)
        self.out_norm = nn.RMSNorm(input_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u is (b, L, d)
        xv = self.in_norm(self.in_proj(u)) # (b, L, 2 * E)
        x, v = xv.chunk(2, dim=-1) # (b, L, E), (b, L, E)
        x_conv = self.conv(x.transpose(-1, -2)).transpose(-1, -2) # (b, L, d)
        gate = v * x_conv # (b, L, d)
        x = self.out_norm(self.out_proj(gate)) # (b, L, d)
        return x


class DropoutNd(nn.Module):
    def __init__(
        self,
        p: float = 0.5,
        tie: bool = True,
        transposed: bool = True,
    ):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), "
                           "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.training:
            if not self.transposed: X = rearrange(X, 'b ... d -> b d ...')
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            X = X * mask * (1.0/(1-self.p))
            if not self.transposed: X = rearrange(X, 'b d ... -> b ... d')
        return X


class S4DKernel(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        N: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr: float | None = None,
    ):
        super().__init__()
        H = hidden_size
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)
        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)
    
    def forward(self, L: int) -> torch.Tensor:
        dt = torch.exp(self.log_dt)
        C = torch.view_as_complex(self.C)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        dtA = A * dt.unsqueeze(-1)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real
        return K
    
    def register(self, name, tensor, lr=None):
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
        optim = {"weight_decay": 0.0}
        if lr is not None: optim["lr"] = lr
        setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        d_state: int = 64,
        dropout: float = 0.0,
        transposed: bool = True,
        **kernel_args,
    ):
        super().__init__()
        self.h = hidden_size
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed
        self.D = nn.Parameter(torch.randn(self.h))
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)
        self.activation = nn.GELU()
        self.dropout = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )
    
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)
        k = self.kernel(L=L)
        k_f = torch.fft.rfft(k, n=2*L)
        u_f = torch.fft.rfft(u, n=2*L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L]
        y = y + u * self.D.unsqueeze(-1)
        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed: y = y.transpose(-1, -2)
        return y


class Lyra(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        pgc_configs: list[tuple[int, float, int]],
        num_s4: int,
        input_size: int,
        dropout: float = 0.2,
        prenorm: bool = True,
    ):
        super().__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.pgc_layers = nn.ModuleList()
        for config in pgc_configs:
            pgc_size, pgc_expansion_factor, num_layers = config
            for _ in range(num_layers):
                self.pgc_layers.append(PGC(
                    hidden_size,
                    pgc_size,
                    pgc_expansion_factor,
                    dropout,
                ))
        
        self.prenorm = prenorm
        
        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(num_s4):
            self.s4_layers.append(
                S4D(hidden_size, dropout=dropout, transposed=True, lr=0.003)
            )
            self.norms.append(nn.RMSNorm(hidden_size))
            self.dropouts.append(nn.Dropout(dropout))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (b, L, input_size) -> (b, L, hidden_size)
        x = self.encoder(x)
        for pgc_layer in self.pgc_layers:
            x = pgc_layer(x)

        # (b, hidden_size, L) -> (b, L, hidden_size)
        x = x.transpose(-1, -2)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            # Apply S4 block
            z = layer(z)
            # Dropout on the output of the S4 block
            z = dropout(z)
            # Residual connection
            x = z + x
            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)
        
        x = x.transpose(-1, -2)
        return x


class LyraConfig(PretrainedConfig):
    model_type = "lyra"
    def __init__(
        self,
        hidden_size: int = 64,
        pgc_configs: list[tuple[int, float, int]] = [(16, 1.0, 1), (128, 1.0, 1)],
        num_s4: int = 1,
        input_size: int = 23, # canonical amino acids + cls, eos and X (20 + 3)
        dropout: float = 0.2,
        prenorm: bool = True,
        num_labels: int = 2,
        task_type: str = 'singlelabel',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.pgc_configs = pgc_configs
        self.num_s4 = num_s4
        self.input_size = input_size
        self.dropout = dropout
        self.prenorm = prenorm
        self.num_labels = num_labels
        self.task_type = task_type 


class LyraForSequenceClassification(PreTrainedModel):
    config_class = LyraConfig
    def __init__(self, config: LyraConfig):
        super().__init__(config)
        self.lyra = Lyra(
            hidden_size=config.hidden_size,
            pgc_configs=config.pgc_configs,
            num_s4=config.num_s4,
            input_size=config.input_size,
            dropout=config.dropout,
            prenorm=config.prenorm,
        )
        classifier_dim = intermediate_correction_fn(2.0, config.num_labels)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, classifier_dim),
            nn.GELU(),
            nn.Linear(classifier_dim, config.num_labels),
        )
        self.loss_fct = get_loss_fct(config.task_type)
        self.num_labels = config.num_labels
        self.task_type = config.task_type

    def _mean_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.mean(dim=1)
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
    ) -> SequenceClassifierOutput:
        x = self.lyra(input_ids)
        x = self._mean_pooling(x, attention_mask)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            if self.task_type == 'regression':
                loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
            elif self.task_type == 'multilabel':
                loss = self.loss_fct(logits, labels.float())
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1).long())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class LyraForTokenClassification(PreTrainedModel):
    config_class = LyraConfig
    def __init__(self, config: LyraConfig):
        super().__init__(config)
        self.lyra = Lyra(
            hidden_size=config.hidden_size,
            pgc_configs=config.pgc_configs,
            num_s4=config.num_s4,
            input_size=config.input_size,
            dropout=config.dropout,
            prenorm=config.prenorm,
        )
        self.loss_fct = get_loss_fct(config.task_type)
        classifier_dim = intermediate_correction_fn(2.0, config.num_labels)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, classifier_dim),
            nn.GELU(),
            nn.Linear(classifier_dim, config.num_labels),
        )
        self.loss_fct = get_loss_fct(config.task_type)
        self.num_labels = config.num_labels
        self.task_type = config.task_type

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
    ) -> TokenClassifierOutput:
        x = self.lyra(input_ids)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            if self.task_type == 'regression':
                loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
            elif self.task_type == 'multilabel':
                loss = self.loss_fct(logits, labels.float())
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1).long())

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


if __name__ == "__main__":
    # py -m probes.lyra_probe
    """
    Standard Lyra includes two PGC blocks
        first hidden dimension of 16
        second of 128
    These are followed by S4D layers, hidden dimension 64, with residual and prenorm
    """
    input_size = 20
    hidden_size = 64
    pgc_configs = [(16, 1.0, 1), (128, 1.0, 1)]
    num_s4 = 1
    
    # Test sequence classification model
    print("\nTesting LyraForSequenceClassification")
    config = LyraConfig(
        hidden_size=hidden_size,
        pgc_configs=pgc_configs,
        num_s4=num_s4,
        input_size=input_size,
        dropout=0.2,
        num_labels=3,
        task_type='singlelabel'
    )
    seq_model = LyraForSequenceClassification(config)
    seq_model.train()
    
    # Forward pass
    batch_size = 2
    seq_length = 50
    x = torch.randn(batch_size, seq_length, input_size)
    attention_mask = torch.ones(batch_size, seq_length)
    labels = torch.randint(0, 3, (batch_size,))
    
    outputs = seq_model(x, attention_mask=attention_mask, labels=labels)
    print(f"Loss: {outputs.loss.item()}")
    print(f"Logits shape: {outputs.logits.shape}")
    
    # Backward pass
    outputs.loss.backward()
    print("Backward pass completed successfully")
    