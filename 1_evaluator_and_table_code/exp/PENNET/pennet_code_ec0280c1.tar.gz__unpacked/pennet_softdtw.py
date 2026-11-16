"""Batched soft-DTW (Cuturi & Blondel 2017) with per-sample lengths, as a custom autograd Function.

The official PEN-Net train.py calls soft_dtw_cuda.SoftDTW(use_cuda=True, gamma=0.1) (Maghoumi's implementation, squared
Euclidean cost, normalize=False; the file is not in the repository) once per sample in a Python loop, on
label[:len_gt, :2] vs pred[:len_pred, :2].  This module computes the same per-sample values for a whole batch at once:
cells beyond a sample's (n_b, m_b) never influence R[n_b, m_b], so the batch DP runs on the padded matrices and
R[b, n_b, m_b] is read out; the backward recursion is seeded at each sample's own terminal cell.

Anti-diagonal ("skewed") layout: S[b, d, i] = R[b, i, d - i] with d = i + j, so each DP step is a slice operation.
Checked against a naive per-sample float64 reference (values and gradients) in test_pennet.py.
"""
import torch

INF = float("inf")


def _skew(D, N, M):
    """D (B, N, M) -> Ds (B, N+M+1, N+1) with Ds[b, d, i] = D[b, i-1, d-i-1] for 1<=i<=N, 1<=d-i<=M (else 0)."""
    B = D.size(0)
    dev = D.device
    d = torch.arange(N + M + 1, device=dev).view(-1, 1)
    i = torch.arange(N + 1, device=dev).view(1, -1)
    j = d - i
    valid = (i >= 1) & (j >= 1) & (j <= M)
    ii = (i - 1).clamp(0, N - 1).expand_as(j)
    jj = (j - 1).clamp(0, M - 1)
    flat = (ii * M + jj).reshape(-1)
    Ds = D.reshape(B, N * M)[:, flat].reshape(B, N + M + 1, N + 1)
    return Ds.masked_fill(~valid, 0.0), valid


def _unskew(Es, N, M):
    """Es (B, N+M+1, N+1) -> G (B, N, M) with G[b, i-1, j-1] = Es[b, i+j, i]."""
    dev = Es.device
    i = torch.arange(1, N + 1, device=dev).view(-1, 1)
    j = torch.arange(1, M + 1, device=dev).view(1, -1)
    return Es[:, i + j, i.expand(N, M)]


class _SoftDTWBatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, D, n, m, gamma):
        B, N, M = D.shape
        Ds, valid = _skew(D, N, M)
        S = D.new_full((B, N + M + 1, N + 1), INF)
        S[:, 0, 0] = 0.0
        for dd in range(2, N + M + 1):
            a = S[:, dd - 1, :-1]          # R[i-1, j]
            b = S[:, dd - 1, 1:]           # R[i, j-1]
            c = S[:, dd - 2, :-1]          # R[i-1, j-1]
            st = torch.stack((a, b, c), dim=0)
            mn = st.min(dim=0).values
            fin = torch.isfinite(mn)
            mn0 = torch.where(fin, mn, torch.zeros_like(mn))
            sm = mn0 - gamma * torch.log(torch.exp(-(st - mn0) / gamma).sum(dim=0))
            val = sm + Ds[:, dd, 1:]
            S[:, dd, 1:] = torch.where(valid[dd, 1:] & fin, val, torch.full_like(val, INF))
        bidx = torch.arange(B, device=D.device)
        out = S[bidx, n + m, n]
        ctx.save_for_backward(S, Ds, n, m)
        ctx.gamma, ctx.NM = gamma, (N, M)
        return out

    @staticmethod
    def backward(ctx, gout):
        S, Ds, n, m = ctx.saved_tensors
        gamma = ctx.gamma
        N, M = ctx.NM
        B = S.size(0)
        E = torch.zeros_like(S)
        bidx = torch.arange(B, device=S.device)
        Tind = torch.zeros_like(S)
        Tind[bidx, n + m, n] = 1.0                 # dL/dR at each sample's own terminal cell (n_b, m_b)
        zero = torch.zeros((), device=S.device, dtype=S.dtype)

        def term(r_s, d_s, e_s, cur):
            ok = torch.isfinite(r_s) & torch.isfinite(cur) & (e_s != 0)
            ex = torch.where(ok, (r_s - d_s - cur) / gamma, zero)
            return torch.where(ok, e_s * torch.exp(ex), zero)

        for dd in range(N + M, 1, -1):
            if dd + 1 <= N + M:
                cur = S[:, dd, 1:]                                   # R[i, j], i = 1..N
                acc = term(S[:, dd + 1, 1:], Ds[:, dd + 1, 1:], E[:, dd + 1, 1:], cur)   # successor (i, j+1)
                acc[:, :-1] += term(S[:, dd + 1, 2:], Ds[:, dd + 1, 2:], E[:, dd + 1, 2:], cur[:, :-1])  # (i+1, j)
                if dd + 2 <= N + M:
                    acc[:, :-1] += term(S[:, dd + 2, 2:], Ds[:, dd + 2, 2:], E[:, dd + 2, 2:], cur[:, :-1])  # (i+1, j+1)
                E[:, dd, 1:] = acc + Tind[:, dd, 1:]
            else:
                E[:, dd, 1:] = Tind[:, dd, 1:]
        G = _unskew(E, N, M) * gout.view(B, 1, 1)
        return G, None, None, None


def sq_euclid(x, y):
    """x (B, N, 2), y (B, M, 2) -> (B, N, M) squared Euclidean cost (as Maghoumi's _euclidean_dist_func)."""
    return ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1)


def soft_dtw_batch(x, y, n, m, gamma=0.1):
    """Per-sample soft-DTW between x[b, :n_b] and y[b, :m_b] (squared Euclidean cost). Returns (B,)."""
    return _SoftDTWBatch.apply(sq_euclid(x, y), n.long(), m.long(), float(gamma))


def soft_dtw_reference(x, y, gamma=0.1):
    """Naive single-pair soft-DTW (plain autograd; for tests only). x (N,2), y (M,2)."""
    D = ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(-1)
    N, M = D.shape
    inf = torch.tensor(INF, dtype=D.dtype)
    R = [[inf] * (M + 1) for _ in range(N + 1)]
    R[0][0] = torch.tensor(0.0, dtype=D.dtype)
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            st = torch.stack((R[i - 1][j], R[i][j - 1], R[i - 1][j - 1]))
            mn = st.min()
            R[i][j] = D[i - 1, j - 1] + mn - gamma * torch.log(torch.exp(-(st - mn) / gamma).sum())
    return R[N][M]
