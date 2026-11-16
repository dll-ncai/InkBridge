"""PEN-Net (Chen et al., ACCV 2022, "Complex Handwriting Trajectory Recovery: Evaluation Metrics and Algorithm").

Architecture transcribed from the official repository github.com/ChenZhounan/PEN-Net @ 03bc5919 (model.py, class Net),
with the released config configs/sym_attn_z_resdtw_pot.yml:
    ENCODER=SymmetricalDualCRNN, BIDIRECTION=True, AttentionEN=True, DECODE_WITHOUT_Z=False
(FORCE_SYM_* flags are not set in the config; with SymmetricalDualCRNN every conv is 3x3 either way, so they have no effect.)

  * CRNN_x (cnn1 + rnn_encoder1): 3x3 convs, asymmetric (2,1) poolings -> 64x64 image -> 1 x 33 feature sequence (256 ch)
    -> 3-layer BiLSTM (128 per direction).
  * CRNN_y (cnn2 + rnn_encoder2): same convs, (1,2) poolings -> 33 x 1 sequence -> 3-layer BiLSTM.
  * Attention (paper eq. 2-4): s_i = U v_i (Linear 256->1), softmax over the 66 concatenated features, Z = sum_i w_i v_i.
  * Decoder init: h0/c0 = Linear(512->256)(concat of the two encoders' final fwd/bwd states, per layer).
  * Global tracing decoder: 3-layer LSTM(256+5 -> 256); input = [p_{i-1}, Z] at every step; output Linear(256->5)
    = (x, y, logit_pen_down, logit_pen_up, logit_eos).

Point representation (re-implementation choice R3, see README_PENNET.md): x, y are ABSOLUTE canvas coordinates in
[0, 64) (paper sec. 5.2 "normalize the online trajectories to [0,64)"; the released PotDataset sets self.absolute=True and
builds labels with corrds2dxdys_abso).  The released decoder_cell() adds predictions to the previous position and rounds
them (a relative-offset test path that contradicts the absolute training labels); here inference feeds back the predicted
absolute (x, y) unchanged (no rounding) and the arg-max one-hot pen state.
"""
import torch
from torch import nn
import torch.nn.functional as F


def _conv(cin, cout):
    return nn.Conv2d(cin, cout, kernel_size=3, padding=(1, 1))


def _cnn(flag):
    if flag == "x":
        pool = lambda: nn.MaxPool2d((2, 1), (2, 1), padding=0)
        last_pad = (0, 1)
    else:
        pool = lambda: nn.MaxPool2d((1, 2), (1, 2), padding=0)
        last_pad = (1, 0)
    return nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=3, padding=(1, 1)), nn.ReLU(), nn.MaxPool2d(2, 2),
        _conv(64, 128), nn.ReLU(), pool(),
        _conv(128, 256), nn.BatchNorm2d(256), nn.ReLU(),
        _conv(256, 256), nn.ReLU(), pool(),
        _conv(256, 256), nn.ReLU(), pool(),
        _conv(256, 256), nn.BatchNorm2d(256), nn.ReLU(),
        _conv(256, 256), nn.ReLU(), pool(),
        nn.Conv2d(256, 256, kernel_size=2, padding=last_pad), nn.BatchNorm2d(256), nn.ReLU())


SOS = (0.0, 0.0, 1.0, 0.0, 0.0)   # official: torch.Tensor(bs * [[[0, 0, 1, 0, 0]]])


class PENNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = _cnn("x")
        self.cnn2 = _cnn("y")
        self.rnn_encoder1 = nn.LSTM(256, 128, bidirectional=True, num_layers=3, batch_first=True)
        self.rnn_encoder2 = nn.LSTM(256, 128, bidirectional=True, num_layers=3, batch_first=True)
        self.fusion = nn.Linear(512, 256)
        self.attention = nn.Linear(256, 1)
        self.rnn_decoder = nn.LSTM(256 + 5, 256, num_layers=3, batch_first=True)
        self.output_layer = nn.Linear(256, 5)

    @staticmethod
    def _bi_state(h, n):
        h = h.view(3, 2, n, 128)
        return torch.cat([h[:, 0], h[:, -1]], dim=-1)          # (3, n, 256): fwd | bwd per layer (official)

    def encode(self, img):
        n = img.size(0)
        x1 = self.cnn1(img)                                     # (n, 256, 1, 33)
        _, c, h, w = x1.size()
        x1 = x1.reshape(n, c, h * w).transpose(1, 2)            # (n, 33, 256)
        x1, (h1, c1) = self.rnn_encoder1(x1)
        x2 = self.cnn2(img)                                     # (n, 256, 33, 1)
        x2 = x2.transpose(2, 3).reshape(n, c, h * w).transpose(1, 2)
        x2, (h2, c2) = self.rnn_encoder2(x2)
        h1, c1, h2, c2 = (self._bi_state(t, n) for t in (h1, c1, h2, c2))
        hn = self.fusion(torch.cat((h1, h2), dim=-1)).contiguous()
        cn = self.fusion(torch.cat((c1, c2), dim=-1)).contiguous()   # official applies the same fusion layer to c
        v = torch.cat((x1, x2), dim=1)                          # (n, 66, 256)
        wgt = F.softmax(self.attention(v), dim=1)
        z = torch.sum(v * wgt, dim=1, keepdim=True)             # (n, 1, 256)
        return z, hn, cn, wgt

    def forward(self, img, seq_in):
        """Teacher-forced training pass. seq_in: (n, T, 5) = [SOS, p_1 .. p_{T-1}]. Returns raw (n, T, 5)."""
        z, hn, cn, _ = self.encode(img)
        t = seq_in.size(1)
        dec_in = torch.cat((seq_in, z.expand(-1, t, -1)), dim=2)
        out, _ = self.rnn_decoder(dec_in, (hn, cn))
        return self.output_layer(out)

    @torch.no_grad()
    def decode_greedy(self, img, max_step):
        """Autoregressive greedy decoding. Returns (n, max_step, 5) rows [x, y, one-hot state] and, per sample, the
        number of emitted rows up to and including the first EOS row (max_step if no EOS was emitted)."""
        z, h, c, _ = self.encode(img)
        n = img.size(0)
        prev = img.new_tensor(SOS).view(1, 1, 5).expand(n, 1, 5).contiguous()
        out = img.new_zeros((n, max_step, 5))
        done = torch.zeros(n, dtype=torch.bool, device=img.device)
        length = torch.full((n,), max_step, dtype=torch.long, device=img.device)
        for i in range(max_step):
            o, (h, c) = self.rnn_decoder(torch.cat((prev, z), dim=2), (h, c))
            o = self.output_layer(o)                            # (n, 1, 5)
            st = torch.argmax(o[:, :, 2:], dim=-1)              # 0 down, 1 up, 2 eos
            row = torch.cat((o[:, :, :2], F.one_hot(st, 3).to(o)), dim=2)
            out[:, i] = row[:, 0]
            newly = (st[:, 0] == 2) & ~done
            length[newly] = i + 1
            done |= newly
            if bool(done.all()):
                break
            prev = row
        return out, length
