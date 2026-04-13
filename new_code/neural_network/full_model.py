import torch
import torch.nn as nn
import torch.nn.functional as F

from new_code.neural_network import model_modules

class MultiModalFusionModel(nn.Module):
    def __init__(
        self,
        modalities: dict,
        n_temporal_layers: int = 3,
        b_mod_early_fusion: bool = True,
        d_encoder_dim: int = 32,
        d_encoder_hidden: int = 128,
        encoder_time_dilation: int = 1,
        encoder_time_kernel: int = 7, 
        b_use_maxpooling_layer: bool = True,
        sequential_model_hidden: int = 128,
        sequential_model_type: str = "lstm",
        sequential_model_layers: int = 1,
        sequential_pooling: str = "attn",
        dropout: float = 0.1,
        outhead: str = "fcl",
        label_dim: int = 3,
        **kwargs
    ):
        super().__init__()

        assert modalities.keys() in ["firing_rate", "lfp_micro", "lfp_macro", "pupil", "gaze"], "unsupported modalities"

        last_window_mode = kwargs.get("last_window_mode", "mean")
        last_window_size = kwargs.get("last_window_size", 5)

        self.d_firing = modalities.get("firing_rate", {}).get("dim", 0)
        self.d_lfp_micro = modalities.get("lfp_micro", {}).get("dim", 0)
        self.d_lfp_macro = modalities.get("lfp_macro", {}).get("dim", 0)
        self.d_gaze = modalities.get("gaze", {}).get("dim", 0)
        self.d_pupil = modalities.get("pupil", {}).get("dim", 0)

        self.d_all_mods = self.d_firing + self.d_lfp_macro + self.d_lfp_macro + self.d_gaze + self.d_pupil
        self.mod_num = len(modalities.keys())

        # -----------------------------
        # Modality encoders
        # -----------------------------
        self.d_encoders = None
        if b_mod_early_fusion:
            self.all_mods_enc = model_modules.ModalityEncoder(
                d_in=self.d_all_mods,
                d_hidden=d_encoder_hidden,
                d_model=d_encoder_dim,
                dilation=encoder_time_dilation,
                k=encoder_time_kernel,
                b_use_maxpooling_layer=True,
                n_temporal_layers=n_temporal_layers,
                dropout=dropout,
            )
            self.firing_rate_enc = None
            self.lfp_macro_enc = None
            self.lfp_micro_enc = None
            self.gaze_enc = None
            self.pupil_enc = None
            self.d_encoders = self.d_all_mods
        else:
            self.all_mods_enc = None
            self.firing_rate_enc = build_encoder(
                self.d_firing, d_encoder_hidden, d_encoder_dim, encoder_time_dilation, encoder_time_kernel, b_use_maxpooling_layer, n_temporal_layers, dropout
            )
            self.lfp_macro_enc = build_encoder(
                self.d_lfp_macro, d_encoder_hidden, d_encoder_dim, encoder_time_dilation, encoder_time_kernel, b_use_maxpooling_layer, n_temporal_layers, dropout
            )
            self.lfp_micro_enc = build_encoder(
                self.d_lfp_micro, d_encoder_hidden, d_encoder_dim, encoder_time_dilation, encoder_time_kernel, b_use_maxpooling_layer, n_temporal_layers, dropout
            )
            self.gaze_enc = build_encoder(
                self.d_gaze, d_encoder_hidden, d_encoder_dim, encoder_time_dilation, encoder_time_kernel, b_use_maxpooling_layer, n_temporal_layers, dropout
            )
            self.pupil_enc = build_encoder(
                self.d_pupil, d_encoder_hidden, d_encoder_dim, encoder_time_dilation, encoder_time_kernel, b_use_maxpooling_layer, n_temporal_layers, dropout
            )
            self.d_encoders = d_encoder_dim * self.mod_num
        
        # -----------------------------
        # Temporal modeling strategy
        # -----------------------------
        self.sequential_model = model_modules.SequentialModel(
            d_in=self.d_encoders,
            d_hidden=sequential_model_hidden,
            model_type=sequential_model_type,
            num_layers=sequential_model_layers,
            dropout=dropout,
            out_pool=sequential_pooling,
            last_window_size=last_window_size,
            last_window_mode=last_window_mode,
        )
        fused_out_dim = self.sequential_model.d_out

        # -----------------------------
        # Out head
        # -----------------------------
        self.face_decode_mode = None
        if outhead == "fcl":
            self.out = model_modules.FullyConnectedHead(fused_out_dim, label_dim, dropout=dropout)
        elif outhead == "face_decode":
            self.face_decode_mode = kwargs.get("face_decode_mode", "logits")
            self.out = model_modules.FaceLabelDecoderHead(fused_out_dim, out_dim=label_dim, dropout=dropout)
        elif outhead == "eye_decode":
            self.out = model_modules.EyeDecoderHead(fused_out_dim, dropout=dropout)
        else:
            raise ValueError(f"Unsupported outhead: {outhead}")

        def forward(self, x: torch.Tensor):
            x_firing_date = x["firing_rate"]
            x_lfp_micro = x["lfp_micro"]
            x_lfp_macro = x["lfp_macro"]

        @staticmethod
        def build_encoder(d_in, d_encoder_dim, d_encoder_hidden, encoder_time_dilation, encoder_time_kernel, b_use_maxpooling_layer, n_temporal_layers, dropout):
            if d_in == 0 or d_in is None:
                return None
            else:
                encoder = model_modules.ModalityEncoder(
                    d_in=d_in,
                    d_hidden=d_encoder_dim,
                    d_model=d_encoder_hidden,
                    dilation=encoder_time_dilation,
                    k=encoder_time_kernel,
                    b_use_maxpooling_layer=b_use_maxpooling_layer,
                    n_temporal_layers=n_temporal_layers,
                    dropout=dropout,
                )
                return encoder
