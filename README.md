# LightweightPeptideGen

Mô hình sinh peptide kháng khuẩn (AMP) *de novo* theo hướng nhẹ và đa phương thức.

> **MAJOR-REVISION AUDIT:** các kết quả số cũ không đủ provenance và không được dùng để
> submit. Cấu hình được khóa cho revision là `config/revision.yaml`; dữ liệu phải được
> dựng lại bằng manifest và homology clusters trước khi chạy
> `run_revision_experiments.sh`.

---

## Ý tưởng của mô hình

Mô hình gồm 4 khối, chạy nối tiếp nhau trong một pipeline huấn luyện 3 giai đoạn:

**1. Generator hợp nhất đa phương thức** — [`peptidegen/models/fusion_generator.py`](peptidegen/models/fusion_generator.py)

Một Transformer decoder gọn (3 lớp, `d_model=128`) sinh chuỗi theo kiểu tự hồi quy. Điểm
khác biệt nằm ở phần *memory* mà decoder cross-attend tới: nó là kết quả hợp nhất của hai
luồng thông tin bằng multi-head cross-attention

```
H_fusion = softmax( (H_seq · W_Q)(H_graph · W_K)ᵀ / √d ) (H_graph · W_V)
```

- `H_seq` — luồng ngữ nghĩa, lấy từ embedding token của **ESM-2** (đóng băng, chỉ dùng
  làm đặc trưng nên không kéo backbone nặng vào autograd).
- `H_graph` — trong warm-up, GATv2 chạy trên adjacency suy ra từ attention của ESM-2;
  trong de novo inference, nó chạy trên đồ thị local cố định của latent memory tokens.
  Không có C-alpha contact graph ở runtime.

Vector nhiễu `z` và vector điều kiện lý hoá 6 chiều `C` cũng được nạp qua chính bộ dựng
memory này, nên đường sinh *de novo* (không có chuỗi đầu vào) và đường teacher-forcing
dùng chung một backbone.

Khi chạy vòng GAN, memory chỉ dựng từ `(z, C)` — ESM-2 **không** chạy lại mỗi bước, nhờ đó
`g_steps` cao vẫn nhanh.

**2. Discriminator CNN** — [`peptidegen/models/discriminator.py`](peptidegen/models/discriminator.py)

CNN đa kernel (3/5/7) với minibatch-stddev để chống mode collapse. Hàm mất mát chọn được
qua config: BCE non-saturating hoặc **WGAN-GP**.

**3. Warm-up bằng MLE** — [`scripts/mle_warmup.py`](scripts/mle_warmup.py)

Trước khi vào GAN, generator được học có giám sát (teacher forcing) với ESM-2 token
embeddings đã cache sẵn. Đây là bước duy nhất mà ESM-2 thực sự "chảy" vào generator, và
checkpoint của nó resume thẳng vào pha GAN.

**4. Tinh chỉnh SCST đa mục tiêu** — [`peptidegen/training/rl.py`](peptidegen/training/rl.py)

Self-Critical Sequence Training với reward cân bằng ba mục tiêu:

```
R = w_stab · stability + w_amp · amp_prob + w_hemo · (1 − hemolysis)
L = − (R(sample) − R(greedy)) · Σ_t log P(y_t)
```

`stability` dùng **reward dạng dải** (band) chứ không đơn điệu — nếu chỉ thưởng "II càng
thấp càng tốt" thì RL sẽ đẩy instability index xuống âm để ăn gian. Bên cạnh đó, hệ số
entropy (`--entropy-coef`) là **bắt buộc phải bật**: thiếu nó SCST sụp đổ đa dạng rất
nhanh (uniqueness tụt, bigram diversity gần 0).

---

## Cấu trúc thư mục

```
peptidegen/
├── data/           Dataset, DataLoader, vocabulary, trích xuất đặc trưng lý hoá
├── models/         Generator (fusion + GRU legacy), Discriminator, wrapper ESM-2
├── training/       GANTrainer, các hàm loss, SCST (rl.py)
├── inference/      PeptideSampler
├── evaluation/     Metrics, độ ổn định, foldability, oracle, tính điều khiển được
└── utils/          I/O, FASTA, vẽ biểu đồ

scripts/            Các bước rời của pipeline (xem bảng bên dưới)
baselines/          HydrAMP, M3-CAD, ESM2-Decoder, PepGraphormer để so sánh
config/config.yaml  Toàn bộ siêu tham số
tests/              Smoke test nhanh
tools/              Tiện ích quản lý dữ liệu / model
train.py            Huấn luyện GAN
generate.py         Sinh peptide từ checkpoint
run_all.sh          Chạy toàn bộ pipeline (Linux/Colab)
run_all.ps1         Bản tương đương cho Windows PowerShell
```

---

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Cài PyTorch đúng phiên bản CUDA của máy bạn trước (xem hướng dẫn ở pytorch.org), rồi mới
cài phần còn lại.

Vài lưu ý về dependency:

- `transformers` là backend chính cho ESM-2 (oracle, foldability, contact graph). Đây là
  cái **bắt buộc**.
- `fair-esm` chỉ cần cho `esm2_embedder.py` (đường legacy) và baseline ESM2-Decoder. Thiếu
  nó thì baseline tự động rơi về BiGRU fallback.
- `torch-geometric` là tuỳ chọn — lớp GATv2 trong repo đã có bản cài đặt dense độc lập.
- `rapidfuzz` chỉ cần khi chạy [`scripts/novelty.py`](scripts/novelty.py).

---

## Dữ liệu

Thư mục `dataset/` **không được đưa lên repo** (khá nặng). Bạn cần tự chuẩn bị và đặt vào
đúng chỗ:

```
dataset/
├── train.csv / train.fasta
├── val.csv   / val.fasta
└── test.csv  / test.fasta
```

Không dùng lại `dataset/train.csv`, `val.csv`, `test.csv` cũ cho kết quả báo cáo.
Revision dùng quy trình hai giai đoạn. Manifest phải khai báo SHA-256 của từng
source, định nghĩa nhãn/evidence và lệnh/phiên bản homology clustering. Giai đoạn
đầu xuất FASTA đã lọc và exact-deduplicate; sau khi chạy MMseqs2, giai đoạn hai
tạo `dataset/rebuilt_2026-09-10/` gồm fixed splits, FASTA, row-level provenance ledger và
`dataset_build_report.json` có hash. Các đường dẫn trong report là tương đối với
thư mục chứa report để toàn bộ artifact vẫn kiểm chứng được sau khi chuyển máy.

```bash
python -m peptidegen.data --manifest config/dataset_manifest.json \
  --prepare-clustering-fasta dataset/clustering_input_2026-09-10.fasta

mmseqs easy-cluster dataset/clustering_input_2026-09-10.fasta dataset/mmseqs_0910 \
  dataset/mmseqs_0910_tmp --min-seq-id 0.4 -c 0.8 --cov-mode 0

python -m peptidegen.data --manifest config/dataset_manifest.json \
  --cluster-tsv dataset/mmseqs_0910_cluster.tsv --cluster-format mmseqs_rep_member \
  --output-dir dataset/rebuilt_2026-09-10
```

Sáu biến điều kiện được tính lại trực tiếp từ chuỗi bằng cùng một implementation:
`instability_index`, `aliphatic_index`, `hydrophobic_moment`, `gravy`,
`charge_at_pH7`, `aromaticity`. Hai proxy cũ `therapeutic_score` và
`hemolytic_score` bị loại vì chưa có định nghĩa/provenance kiểm chứng được.

---

## Chạy nhanh để kiểm tra

Trước khi chạy thật, nên chạy thử với dữ liệu cắt nhỏ để chắc chắn không có gì gãy:

```bash
python scripts/mle_warmup.py --config config/revision.yaml --conditional \
  --esm-model esm2_t6_8M_UR50D --epochs 1 --max-samples 2000 --out checkpoints/warmup.pt

python train.py --config config/revision.yaml --conditional \
  --resume checkpoints/warmup.pt --epochs 1 --batch-size 32 --max-samples 2000

python generate.py --checkpoint checkpoints/best_model.pt --model-id exploratory \
  --num 50 -o test.fasta
```

Hoặc dùng luôn bộ test có sẵn:

```bash
pytest tests/ -q
```

---

## Chạy toàn bộ pipeline

```bash
# Linux / Colab
PYTHON_BIN=python bash run_revision_experiments.sh

# Windows
.\run_all.ps1
```

Script revision yêu cầu dataset build report và hai reward-oracle checkpoint đã audit,
sau đó chạy full model cùng ma trận ablation trên 5 training seeds. External predictors
và ESMFold chạy/import riêng vì cần phần mềm/dịch vụ và GPU khác.

Các biến môi trường điều chỉnh được: `ESM`, `EPOCHS`, `GSTEPS`, `BATCH`, `NUM_GEN`,
`SCST_STEPS`, `WARMUP_EPOCHS`.

Nếu chạy trên Google Colab, xem [`COLAB_GUIDE.md`](COLAB_GUIDE.md) — có sẵn từng cell để
dán vào notebook, kèm cách trỏ `checkpoints/` sang Drive để resume khi hết phiên.

---

## Chạy từng bước

| Bước | Lệnh | Việc nó làm |
|---|---|---|
| Warm-up | `scripts/mle_warmup.py --conditional --esm-model ... --contact-graph` | Học có giám sát với ESM-2 embeddings + đồ thị tiếp xúc |
| GAN | `train.py --conditional --resume checkpoints/warmup.pt` | Huấn luyện đối kháng |
| Oracle AMP | `scripts/train_oracle.py amp --train ... --test ...` | ESM-2 + logistic regression, báo AUC trên test |
| Oracle hemolysis | `scripts/train_oracle.py hemo --train ... --test ...` | Cần nhãn ngoài (Hemolytik2, tải 2026-09-10; endpoint hồng cầu người, ngưỡng 100 µM) |
| SCST | `scripts/scst_finetune.py --checkpoint ... --amp-oracle ... --entropy-coef 0.02` | Tinh chỉnh RL đa mục tiêu |
| Sinh chuỗi | `generate.py --checkpoint checkpoints/scst_model.pt --model-id full --num 1000 --seed 42` | Xuất FASTA + provenance sidecar |
| Đánh giá | `scripts/evaluate_generated.py --gen-dir ... --train-fasta dataset/rebuilt_2026-09-10/train.fasta --sequence-plausibility` | Gộp nhiều seed, báo mean±SD/CI + kiểm định thống kê |
| Điều khiển được | `scripts/controllability.py --checkpoint ... --features ...` | Quét giá trị mục tiêu, đo Spearman/Pearson/MAE |
| Độ mới | `scripts/novelty.py --gen-dir results/gen` | So khớp chính xác + khoảng cách Levenshtein với tập train |
| Đếm tham số | `scripts/backbone_ablation.py params` | Báo số tham số hoạt động thực tế |
| Xuất bảng paper | `scripts/export_verified_manuscript_tables.py --paper-dir ...` | Chỉ mở khóa bảng khi toàn bộ artifact audit đã pass |

Hai script chạy trên Colab/GPU lớn: [`scripts/esmfold_plddt.py`](scripts/esmfold_plddt.py)
(pLDDT bằng ESMFold) và [`scripts/esmfold_contacts.py`](scripts/esmfold_contacts.py)
(đồ thị tiếp xúc Cα < 8 Å chuẩn, thay cho attention contacts của ESM-2).

---

## So sánh với baseline

Các module `hydramp` và `m3cad` trong repo chỉ là **architecture-inspired controls**,
không phải reproduction chính thức; riêng M3CAD-inspired không có nhánh voxel 3D của
bài báo. Không được ghi tên chúng là HydrAMP/M3-CAD trong bảng so sánh trực tiếp. Muốn
claim official retraining phải chạy repository/weights chính thức và lưu commit, môi
trường, preprocessing, checkpoint-selection và raw outputs.

```bash
python baselines/train_baseline.py --model hydramp --epochs 50
python scripts/gen_baseline.py --model hydramp --name HydrAMP \
  --checkpoint <ckpt> --num 1000 --out-dir results/gen
python scripts/evaluate_generated.py --gen-dir results/gen --amp-oracle results/oracle_amp.pkl
```

Thay `hydramp` bằng `m3cad` hoặc `esm2gen` cho hai baseline còn lại.

> AMP reward-oracle score chỉ là training diagnostic. Independent AMP, general toxicity
> và hemolysis-specific prediction phải được import bằng manifest và không được dùng
> trong training, model selection hoặc tuning.

---

## Vài tham số đáng chú ý trong `config/config.yaml`

| Khoá | Ý nghĩa |
|---|---|
| `model.architecture` | `fusion` (mặc định, đúng bài báo) hoặc `gru` (legacy, chỉ để ablation) |
| `model.esm_dim` | Phải khớp chiều ESM-2 dùng ở warm-up: `480` cho 35M, `1280` cho 650M, `null` nếu không dùng |
| `training.gan_loss` | `bce` hoặc `wgan_gp` |
| `training.discrete_relaxation` | `softmax` hoặc `gumbel` (straight-through, gradient sạch hơn) |
| `training.batch_size` | **Hạ xuống ~64** khi dùng generator fusion — giá trị mặc định to là di sản của kiến trúc GRU cũ và sẽ gây OOM |
| `training.g_steps` | Config để 10; đặt 4 nhanh hơn nhiều mà chất lượng gần như không đổi |

---

## Yêu cầu phần cứng

Nhẹ hơn nhiều so với cảm giác ban đầu — điểm chính của hướng tiếp cận này:

- Toàn bộ pipeline chạy được trên **một GPU đơn**, đỉnh VRAM khoảng **2 GB** ở batch 64.
- Chạy được cả trên laptop GPU 8 GB (đã kiểm chứng), Colab T4, hoặc GPU workstation.
- Phần tốn thời gian nhất là **sinh chuỗi tự hồi quy** trong vòng GAN, tỉ lệ thuận với
  `g_steps` — muốn nhanh thì giảm `g_steps` trước, đừng giảm epoch.
- Backbone ESM-2 650M cần thêm bộ nhớ lúc warm-up; nếu chật thì dùng 35M, chất lượng
  giảm không đáng kể.

---

## Ghi chú khi chạy trên Windows

- Config có comment tiếng Việt nên loader mở file bằng UTF-8; đừng đổi encoding.
- `peptidegen/__init__.py` ép stdout sang UTF-8 để log có ký tự ✓ không làm crash console
  cp1252.
- `run_all.ps1` truyền tham số dạng mảng, không phải chuỗi có dấu cách — giữ nguyên nếu
  bạn sửa script.

---

## Giấy phép

Mã nguồn trong repo này phát hành theo **MIT License** — xem [`LICENSE`](LICENSE).

Giấy phép đó **chỉ phủ mã nguồn của repo này**. Các cơ sở dữ liệu và công cụ bên ngoài mà
công trình phụ thuộc đều được **trích dẫn và dẫn link, không đóng gói lại** ở đây, và điều
khoản riêng của chúng vẫn hiệu lực — xem [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Bản lưu bất biến của mã đã tạo ra các kết quả được báo cáo:
DOI [10.5281/zenodo.22771618](https://doi.org/10.5281/zenodo.22771618),
tương ứng tag `v1.0.0` (commit `624c130`). Các commit sau đó trên `main` chứa bản sửa công
cụ tìm ra sau khi các lần chạy đã xong; từng bản sửa được liệt kê kèm lý do nó không thể
đổi con số nào trong [`CHANGELOG_POST_RELEASE.md`](CHANGELOG_POST_RELEASE.md).

Ai muốn **tái lập** số liệu đã báo cáo thì dùng bản lưu ở DOI; ai muốn **phát triển tiếp**
thì dùng `main`.
