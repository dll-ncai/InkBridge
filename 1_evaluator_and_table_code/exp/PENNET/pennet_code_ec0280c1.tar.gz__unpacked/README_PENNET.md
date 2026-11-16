# PENNET: PEN-Net re-implementation on OUHD-L v3 ligature units

## Identification
* Model: PEN-Net, "Parsing-and-tracing ENcoder-decoder Network".
* Paper: Zhounan Chen, Daihui Yang, Jinglin Liang, Xinwu Liu, Yuyi Wang, Zhenghua Peng, Shuangping Huang.
  *Complex Handwriting Trajectory Recovery: Evaluation Metrics and Algorithm.* ACCV 2022 (Asian Conference on Computer
  Vision), Springer LNCS 13842, pp. 58-74, doi:10.1007/978-3-031-26284-5_4; CVF open access pp. 1060-1076; arXiv:2210.15879.
  The same paper introduces the AIoU and LDTW metrics.
* Architecture (paper sec. 4): a double-stream parsing encoder (CRNN_x and CRNN_y, each a CNN with asymmetric pooling plus
  a 3-layer BiLSTM), an attention fusion giving the glyph-parsing feature Z, and a global tracing decoder (3-layer LSTM
  whose input at every step is [p_{i-1}, Z], initialised from the encoder states). Output per step: (x, y) plus a 3-way
  pen state (down / up / end-of-sequence).
* Losses (paper sec. 4.3): L = 0.5 L1 + 1 weighted CE (weights down 1, up 5, eos 1) + (1/6000) SDTW.
* Training (paper sec. 5.2): trajectories normalised to [0,64); images rendered from online data; 500k iterations
  (Chinese, Japanese) or 200k (English, Indic); batch 512; Adam, lr 1e-3; one RTX3090.
* Data (paper sec. 5.1): CASIA-OLHWDB1.1 Chinese; English letters from CASIA-OLHWDB 1.0-1.2 (52 classes);
  Nakayosi_t-98-09 / Kuchibue_d-96-02 Japanese; HP Labs Tamil (156 classes).
* Reported (paper Table 1; AIoU / LPIPS / LDTW / DTW / RMSE): Chinese 0.450 / 0.113 / 3.11 / 233.8 / 14.39;
  English 0.461 / 0.074 / 3.21 / 77.35 / 15.12; Indic 0.546 / 0.074 / 1.62 / 104.9 / 12.92;
  Japanese 0.476 / 0.125 / 3.39 / 144.5 / 17.08; Chinese complex 0.411 / 0.143 / 3.58 / 402.9 / 15.61;
  Japanese complex 0.445 / 0.186 / 2.95 / 344.4 / 16.00.

## Official code: present but not runnable
github.com/ChenZhounan/PEN-Net at commit 03bc59199f47d34e6df09b1ecdb956bbc284bd4a (2022-10-03), cloned to
$U/repos/PEN-Net. It contains README.md, configs/sym_attn_z_resdtw_pot.yml, model.py (class Net), train.py,
datasets/__init__.py and datasets/casia.py. It has no weights and no licence file.
It cannot run as released:
* train.py builds `model.Net_tr_2d`, which model.py does not define.
* config.py, datasets/data_utils.py, soft_dtw_cuda.py and visualization.py are missing.
* The iam, hindi and japanese loaders that datasets/__init__.py imports are missing.

So this run is a re-implementation. model.Net, the only model in the repository, is used as the architecture reference.
No third-party PEN-Net code or weights were found. Published re-runs (Nagamatsu et al.; Wang et al. 2609.02251) say they
re-implemented it or copied Table 1.

## Where the paper and the released code disagree (paper followed unless noted)
| id | item | paper | released code | used |
|---|---|---|---|---|
| C1 | point loss | L1 | config PTS_LOSS_TYPE l2_sqrt | L1 |
| C2 | CE class weights | down 1, up 5, eos 1 | [1, 5, 10] | 1, 5, 1 |
| C3 | learning rate | 1e-3 | 1e-4 | 1e-3 |
| C4 | batch | 512 | 128 per GPU (README launches 2 GPUs) | 512 on 1 GPU |
| C5 | iterations | 500k zh/ja, 200k en/ta | MAX_ITER 4,000,000 | 100k (E0 budget rule, cheaper than either paper schedule) |
| C6 | model class | - | train.py: Net_tr_2d (absent) | model.Net |
| C7 | inference coords | "position" | decoder_cell adds rounded offsets to the previous point (relative), while PotDataset.absolute=True trains on absolute labels | absolute, fed back unrounded (R3) |

## Details the paper leaves open, taken from the released code
| id | item | value |
|---|---|---|
| U1 | CNN per stream | conv3x3 64 + ReLU + maxpool 2x2; conv 128 + ReLU + asym pool; conv 256 + BN + ReLU; conv 256 + ReLU + asym pool; conv 256 + ReLU + asym pool; conv 256 + BN + ReLU; conv 256 + ReLU + asym pool; conv 2x2 (pad (0,1) / (1,0)) + BN + ReLU. Gives 1x33 (x stream) and 33x1 (y stream) |
| U2 | RNN sizes | BiLSTM 3 layers, 128 per direction; decoder LSTM 3 layers, 256; input 256 + 5 |
| U3 | decoder init | h0 = Linear(512->256)(concat per-layer fwd/bwd final states of both encoders); c0 through the same layer |
| U4 | attention | Linear(256->1) score, softmax over the 66 concatenated features, weighted sum (paper eq. 2-4) |
| U5 | SOS token | [0, 0, 1, 0, 0] |
| U6 | SDTW | gamma 0.1, squared Euclidean cost on (x, y), per sample, label[:len_gt] incl. EOS row vs teacher-forced pred[:len_pred], len_pred = first row with arg-max state EOS, + 1 (else T); mean over batch; /6000 |
| U7 | CE reduction | weighted CE with reduction 'none', then mean over all B*T padded rows (padding rows are EOS) |
| U8 | L1 reduction | (abs dx + abs dy) on rows whose state is not EOS, summed, divided by their count |
| U9 | LR schedule | constant (the MultiStepLR object is never attached or stepped) |
| U10 | grad clipping | global L2 norm 0.2 |
| U11 | image normalisation | white ink on black, (v - 127) / 127 |
| U12 | labels | absolute coordinates (PotDataset.absolute = True) |
| U13 | padding | batch padded to its longest sample with rows [0,0,0,0,1]; drop_last |

## Details neither specifies: choices made here
| id | item | choice |
|---|---|---|
| R1 | data | E0 arrays data/ouhdl_e0_v3 (built by experiments/e0/build_e0_data.py). Targets are E0's: per-stroke 1-px arc-length resampling on the 64x64 evaluation canvas, step grown until the unit has at most 320 points. The paper's dedup + 2x downsampling is not applied |
| R2 | input image | E0 cond_rendered (uhwr renderer, 2 px, anti-aliased) for training; cond_rendered and cond_dataset at test. No random thickness 1-3 augmentation (the released CASIA loader uses it; E0 does not) |
| R3 | inference | greedy: arg-max pen state as a one-hot plus predicted absolute (x, y) fed back unrounded; no skeleton snapping (sk_points None); at most 321 steps; stop at the first EOS |
| R4 | pen-state semantics | down = point followed by a connected point; up = last point of a stroke; eos = a separate terminal row (sketch-rnn [34], cited by the paper) |
| R5 | EOS row coordinates | the last point's (x, y) (data_utils.corrds2dxdys_abso is missing) |
| R6 | degenerate outputs | EOS first: one point at the EOS row's (x, y) (counted as n_empty); no EOS within 321 steps: first 320 rows (counted as n_no_eos / n_truncated). No coordinate clipping |
| R7 | training data | the whole v3 train split (33,615 units); val is not used for training. E0 trained on 99% (--train-fraction 0.99) |
| R8 | weights averaging | none (PEN-Net has no EMA; E0 used EMA weights) |
| R9 | determinism | cudnn.deterministic, benchmark off, torch.use_deterministic_algorithms(warn_only); seed 20261016 (E0's); epoch permutation rng(seed + epoch) |
| R10 | selection | as E0: milestones every 10k steps (10 milestones); lowest mean native LDTW on 1,000 random val units (subset seed 0, rendered); ties broken by higher AIoU |
| R11 | evaluation | test (4,406) and gold (354) units, separately, rendered and dataset input; scored by e0_score.py (unchanged copy of e0/code_4cd40f57) with uhwr_eval 0.3.1; both protocols; 95% writer-cluster bootstrap CIs (B=2000, seed 0). e0_score's line-cluster summary also written |
| R12 | decode scope | test/gold predictions are decoded for all 10 milestones on the GPU; only the val-selected milestone is scored |
| R13 | hardware | 1 GPU, compute capability >= 8.0 (A30/A40/A100/H100 nodes; V100, MIG A30 slices, A2, A4000, GTX 10xx excluded by node list) |
| R14 | resume | ckpt_latest.pt every 1k steps (model, optimiser, step, data position, RNG); SIGTERM saves it at the end of the step; Slurm --requeue |

## Not run (next steps)
* IAM component arm (data/d2_v2/iam).
* Validation of this re-implementation against the paper's Table 1 (CASIA-OLHWDB1.1 / HPL Tamil).
