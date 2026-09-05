# Chạy ma trận thí nghiệm trên FITLAB

Soạn 2026-09-05. Dùng sau khi bộ dữ liệu đã dựng xong ở máy local.

Máy local (RTX 4060 Laptop 8,6 GB) đã làm xong phần dữ liệu và oracle. Phần còn lại là
**12 biến thể × 5 seed = 60 lần huấn luyện**, quá nặng cho laptop nên phải đưa lên FITLAB
(Quadro RTX 6000 24 GB).

---

## 1. Chuẩn bị gói mang lên

Ở máy local, nén repo **loại trừ** những thứ nặng và sinh lại được:

```bash
cd "d:/Project/AI Engineer/LightweightPeptideGen"
tar --exclude='.git' --exclude='results/timing' --exclude='results/esm_cache*' \
    --exclude='__pycache__' --exclude='dataset/train.csv' --exclude='dataset/val.csv' \
    --exclude='dataset/test.csv' --exclude='dataset/amps_pdb' \
    -czf lpg_fitlab.tar.gz LightweightPeptideGen-main
```

Phải mang theo (đây là thứ không sinh lại được trên server):

- `dataset/rebuilt_interim/` — bộ dữ liệu đã dựng, kèm `dataset_build_report.json`
- `dataset/derived_from_legacy/` — sáu file nguồn đã curation
- `config/dataset_manifest.interim.json`
- `results/oracles/amp/` — oracle đã huấn luyện

> **Lưu ý:** `provenance_ledger.csv` nặng 202 MB. Nếu đường truyền chậm thì bỏ lại,
> nó không cần cho huấn luyện, chỉ cần khi làm hồ sơ phát hành cuối cùng.

---

## 2. Dựng môi trường trên FITLAB

Theo cách bạn vẫn làm, đặt venv ở `/home/coder` để dùng chung giữa các node:

```bash
cd /home/coder
tar -xzf lpg_fitlab.tar.gz
python3 -m venv .venv
~/.venv/bin/python -m pip install --upgrade pip
# torch dung CUDA cua server truoc
~/.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
~/.venv/bin/python -m pip install -r LightweightPeptideGen-main/requirements-revision.txt
```

Kiểm tra:

```bash
cd /home/coder/LightweightPeptideGen-main
~/.venv/bin/python -m pytest tests -q          # phai 23/23 dat
~/.venv/bin/python -c "import torch;print(torch.cuda.get_device_name(0))"
```

MMseqs2 trên Linux (chỉ cần nếu dựng lại dữ liệu trên server):

```bash
wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
tar xzf mmseqs-linux-avx2.tar.gz
export PATH=$PATH:$(pwd)/mmseqs/bin
```

---

## 3. Commit trước khi chạy — bắt buộc

Runbook quy định cây git phải sạch, nếu không mọi artifact bị đánh dấu không dùng được:

```bash
git status --short        # phai khong in ra gi
git rev-parse HEAD
```

Đây chính là lý do oracle chạy ở local đang có `reportable: false`.

---

## 4. Chạy ma trận

```bash
cd /home/coder/LightweightPeptideGen-main
PYTHON_BIN=~/.venv/bin/python \
ESM2_REVISION=6fbf070e65b0b7291e7bbcd451118c216cff79d8 \
WARMUP_EPOCHS=10 GAN_EPOCHS=50 SCST_STEPS=2000 NUM_GEN=1000 \
nohup bash run_revision_experiments.sh > run_matrix.log 2>&1 &
```

Chạy trong `nohup` vì phiên SSH có thể đứt. Theo dõi bằng `tail -f run_matrix.log`.

### Ngân sách thời gian

Đo được ở local: 0,105 giây cho mỗi (mẫu × epoch) ở `g_steps=4`, trên RTX 4060.
Tập train AMP có 11.009 chuỗi.

| Kịch bản | Một lần chạy | 60 lần |
|---|---|---|
| `g_steps=10` (đúng config) trên 4060 | ~40 giờ | ~100 ngày |
| `g_steps=10` trên RTX 6000, ước nhanh gấp ~3 | ~13 giờ | **~33 ngày** |
| `g_steps=4` trên RTX 6000 | ~5 giờ | **~13 ngày** |

Ngay cả phương án nhanh nhất cũng vượt hạn 17/09. Nên:

1. **Chạy biến thể `full` trước, đủ 5 seed.** Đây là kết quả chính, khoảng 2–3 ngày.
2. Sau đó chạy ba biến thể trả lời trực tiếp Reviewer 1 ý 4: `no_esm2`, `no_gatv2`,
   `concat_fusion`. Thêm khoảng 8 ngày.
3. Các biến thể còn lại chạy sau, trong thời gian gia hạn.

Runbook cấm hạ `g_steps`, epoch hay số seed cho kết quả đưa vào bài. Nếu buộc phải hạ để kịp,
phải khai báo minh bạch trong bài là đã đổi giao thức.

---

## 5. ESMFold, sau khi có FASTA của biến thể `full`

```bash
mkdir -p results/esmfold
for seed in 42 123 456 789 1337; do
  ~/.venv/bin/python scripts/esmfold_plddt.py \
    --input "results/ablations/full/gen/full_seed${seed}.fasta" \
    --output "results/esmfold/full_seed${seed}.csv" \
    --seed "$seed" --model facebook/esmfold_v1 \
    --model-revision 75a3841ee059df2bf4d56688166c8fb459ddd97a
done
~/.venv/bin/python scripts/aggregate_esmfold.py --csv-dir results/esmfold \
  --fasta-dir results/ablations/full/gen --expected-models full \
  --out results/esmfold_summary.json --tex-out results/esmfold_summary.tex
```

ESMFold cần nhiều VRAM. Trên 24 GB nên chạy được, nếu OOM thì giảm batch hoặc lọc bớt chuỗi dài.

---

## 6. Mang kết quả về

Chỉ cần các file JSON và FASTA, không cần checkpoint:

```bash
tar -czf results_back.tar.gz results/ablations/*/benchmark.json \
    results/ablations/*/gen/*.fasta results/esmfold_summary.json \
    results/ablation_study_summary.json results/controllability_summary.json
```

Sau đó ở local chạy `python scripts/audit_artifacts.py` để xem còn thiếu gì.

---

## 7. Việc vẫn cần con người, không server nào thay được

1. **File nguồn gốc thật.** Sáu file trong `dataset/derived_from_legacy/` là bản khôi phục từ
   CSV cũ, không phải bản tải gốc. Trước khi nộp phải thay bằng bản tải thật từ DRAMP, dbAMP,
   DBAASP, APD và AMPBenchmark, kèm phiên bản, ngày tải, giấy phép và hash.
2. **Phê duyệt chính sách curation** đã áp dụng cho 186 cụm lẫn nhãn.
3. **Dữ liệu hemolysis.** Chưa có oracle hemolysis, cần nhãn ngoài kèm định nghĩa endpoint.
4. **Predictor AMP độc lập** để thay cho oracle nội bộ khi báo cáo kết quả kháng khuẩn.
5. **Quyết định về baseline**, xem mục 4 của `TRA_LOI_PHAN_BIEN.md`.
