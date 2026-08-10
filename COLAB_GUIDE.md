# Chạy full pipeline trên Google Colab (GPU lớn)

Hướng dẫn chạy `run_all.sh` trên Colab với ESM-2 650M (đúng paper). Mỗi khối dưới
là **một cell** — dán vào notebook Colab theo thứ tự.

> **Trước tiên:** Runtime → Change runtime type → **GPU** (T4 đủ chạy; A100/L4 nhanh
> hơn nhiều — nên dùng nếu có Colab Pro).
>
> Mã nguồn thì `git clone` thẳng từ GitHub là xong, nhưng **`dataset/` không nằm trong
> repo** nên vẫn cần Drive để đưa dữ liệu lên. Cell 3 dưới đây dùng cách zip-qua-Drive
> cho cả hai — vừa gọn vừa resume được khi hết phiên.

---

### Cell 1 — Kiểm tra GPU
```python
!nvidia-smi
```

### Cell 2 — Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 3 — Đưa project lên Drive (làm 1 lần)
> Trên máy bạn: nén thư mục `LightweightPeptideGen-main` thành `LightweightPeptideGen.zip`
> **nhưng LOẠI TRỪ** `results/`, `checkpoints/`, `__pycache__` (chúng chứa hàng nghìn
> file cache .npy, làm zip nặng vô ích). Upload `.zip` vào Drive
> (vd `MyDrive/LPG/LightweightPeptideGen.zip`). Cell này giải nén + **tự dò đúng thư
> mục chứa run_all.sh** rồi `cd` vào (tránh lỗi "No such file or directory"):
```python
import os, zipfile, glob
os.makedirs('/content/LPG', exist_ok=True)
with zipfile.ZipFile('/content/drive/MyDrive/LPG/LightweightPeptideGen.zip') as z:
    z.extractall('/content/LPG')
# tự tìm thư mục chứa run_all.sh (dù zip có lồng thêm 1 cấp)
hits = glob.glob('/content/LPG/**/run_all.sh', recursive=True)
assert hits, "Khong tim thay run_all.sh trong zip!"
root = os.path.dirname(hits[0])
print('PROJECT ROOT =', root)
%cd $root
!ls run_all.sh dataset/train.csv config/config.yaml   # phai thay du 3 file
```

### Cell 4 — Cài dependency
```python
# transformers cho ESM-2/ESMFold; còn lại Colab đã có sẵn (torch, sklearn, scipy, pandas)
!pip -q install -U "transformers>=4.44" scipy scikit-learn pandas
import torch, transformers
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('transformers', transformers.__version__)
```

### Cell 5 — Lưu checkpoint vào Drive (để resume nếu hết phiên)
```python
import os
os.makedirs('/content/drive/MyDrive/LPG/checkpoints', exist_ok=True)
os.makedirs('/content/drive/MyDrive/LPG/results', exist_ok=True)
# trỏ checkpoints/ và results/ sang Drive
for d in ['checkpoints', 'results']:
    if os.path.islink(d) or os.path.exists(d):
        import shutil; shutil.rmtree(d, ignore_errors=True)
    os.symlink(f'/content/drive/MyDrive/LPG/{d}', d)
print('checkpoints/ và results/ đã trỏ sang Drive')
```

### Cell 6 — Chạy full pipeline (ESM-2 650M)
```python
import os
os.environ['ESM']    = 'esm2_t33_650M_UR50D'   # đúng paper (1280-d); đổi 35M nếu muốn nhanh
os.environ['EPOCHS'] = '40'    # GAN epochs (tăng nếu còn thời gian; có early-stopping)
os.environ['GSTEPS'] = '4'     # config gốc 10; 4 nhanh hơn nhiều, chất lượng tương đương
os.environ['BATCH']  = '64'    # T4 16GB; A100 có thể tăng 128-256
os.environ['NUM_GEN']= '1000'
!bash run_all.sh
```

> ⏱️ **Thời gian:** sinh autoregressive là phần nặng nhất. Với T4 + EPOCHS=40 +
> GSTEPS=4, ước tính vài giờ. Nếu Colab ngắt phiên giữa chừng, chạy lại **Cell 6**
> — `train.py` tự resume từ `checkpoints/best_model.pt` (đã lưu trên Drive).
> Muốn paper-grade hơn thì tăng EPOCHS (60-100) và dùng A100.

### Cell 7 — (tùy chọn) Hemolysis oracle + ESMFold contacts
```python
# Hemolysis oracle: cần file nhãn ngoài (HemoPI/DBAASP) dạng cột sequence,label
# !python scripts/train_oracle.py hemo --train data/hemopi_train.csv --test data/hemopi_test.csv \
#     --model esm2_t33_650M_UR50D --out results/oracle_hemo.pkl --batch-size 64

# Contact <8A chuẩn từ ESMFold (thay cho ESM-2 attention):
# !python scripts/esmfold_contacts.py --input dataset/train.fasta \
#     --output results/esmfold_contacts.npz --radius 8.0 --max-seqs 5000

# Ablation backbone-size (tải 4 model ESM):
# !python scripts/backbone_ablation.py oracle --train dataset/train.csv \
#     --test dataset/test.csv --out results/backbone_oracle.json
```

### Cell 8 — Xem số liệu
```python
import json, glob
for f in sorted(glob.glob('results/*.json')):
    print('\n===', f, '===')
    print(json.dumps(json.load(open(f)), indent=2)[:2000])
```

Kết quả nằm trong `results/` (đã lưu Drive): `benchmark.json` (metric của corpus sinh ra
+ foldability + kiểm định thống kê), `controllability.json` (tương quan giữa giá trị mục
tiêu và giá trị đạt được), và `oracle_amp_report.json` (AUC/ACC/MCC của oracle trên test).
