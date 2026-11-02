# Nghiên cứu và xây dựng mô hình học sâu cho bài toán chuẩn đoán u não từ ảnh MRI

## 1. Tính cấp thiết của đề tài
U não là một bệnh lý nguy hiểm, việc phát hiện sớm có ý nghĩa quan trọng trong nâng cao hiệu quả điều trị và tỷ lệ sống của bệnh nhân. Tuy nhiên, quá trình chuẩn đoán hiện nay chủ yếu phụ thuộc vào kinh nghiệm của bác sĩ và tốn nhiều thời gian khi phân tích ảnh y khoa. 

Trong bối cảnh đó, sự phát triển của các phương pháp học sâu trong xử lý ảnh mở ra khả năng tự động hóa và nâng cao độ chính xác trong chuẩn đoán. Vì vậy, việc nghiên cứu và xây dựng mô hình học sâu cho bài toán này có ý nghĩa khoa học và tiềm năng ứng dụng thực tiễn cao, đồng thời phù hợp với chiến lược phát triển khoa học, công nghệ và đổi mới sáng tạo.

## 2. Mục tiêu nghiên cứu
- Nghiên cứu các phương pháp học sâu trong phân tích ảnh y khoa phục vụ chuẩn đoán u não.
- Xây dựng và huấn luyện các mô hình học sâu cho bài toán phát hiện và phân loại u não từ ảnh MRI.
- Đánh giá hiệu năng các mô hình dựa trên các chỉ số phù hợp và dữ liệu thực nghiệm.
- Đề xuất mô hình có hiệu quả cao, có khả năng hỗ trợ trong bài toán chuẩn đoán thực tế.

## 3. Nội dung nghiên cứu
- **Khảo sát tổng quan:** Tìm hiểu về bệnh lý u não, các kỹ thuật chẩn đoán hình ảnh và xu hướng ứng dụng học sâu trong phân tích ảnh y khoa.
- **Xử lý dữ liệu:** Thu thập, tiền xử lý và chuẩn hóa dữ liệu ảnh MRI phục vụ cho quá trình huấn luyện và đánh giá mô hình.
- **Xây dựng mô hình:** Triển khai và thử nghiệm một số kiến trúc học sâu tiêu biểu cho bài toán phân tích ảnh, tập trung vào các mô hình mạng nơ-ron tích chập (CNN) và các biến thể cải tiến.
- **Huấn luyện và Đánh giá:** Huấn luyện và đánh giá các mô hình trên bộ dữ liệu thực nghiệm thông qua các chỉ số như độ chính xác (Accuracy), độ nhạy (Sensitivity/Recall), độ đặc hiệu (Specificity) và F1-score.
- **Phân tích và So sánh:** Phân tích, so sánh kết quả giữa các mô hình nhằm xác định phương pháp phù hợp nhất cho bài toán.

## 4. Kết quả nghiên cứu dự kiến
- **Mô hình tối ưu:** Hoàn thiện nghiên cứu, xây dựng và đánh giá một số mô hình học sâu. Trên cơ sở kết quả thực nghiệm, đề xuất mô hình phù hợp, đảm bảo hiệu năng tốt và có khả năng tổng quát hóa cao.
- **Quy trình chuẩn hóa:** Xây dựng được quy trình xử lý dữ liệu và huấn luyện mô hình hoàn chỉnh, bao gồm các bước tiền xử lý, huấn luyện và đánh giá.
- **Đóng góp khoa học:** Kết quả nghiên cứu mang tính khảo sát, so sánh và đánh giá hiệu quả của các mô hình. Hoàn thiện báo cáo và kết quả nghiên cứu hướng tới phát triển thành bài báo khoa học hoặc báo cáo tại hội nghị chuyên ngành.

---

## Reproduce (chạy trên máy có GPU)

### 1. Clone & tạo môi trường
```bash
git clone <repo> && cd Brain_tumor

# Linux/macOS
python -m venv .venv && source .venv/bin/activate
# Windows
python -m venv .venv && .venv\Scripts\activate

pip install -r requirements.txt
```

> **GPU:** `requirements.txt` cài torch CPU. Trên máy có CUDA, cài thêm bản CUDA tương ứng, ví dụ CUDA 12.1:
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

### 2. Chuẩn bị dữ liệu

**Cách A — Kaggle API** (cần `~/.kaggle/kaggle.json`):
```bash
python scripts/download_dataset.py   # tải Kaggle dataset -> data/raw
```

**Cách B — tải thủ công bằng browser** (tài khoản Kaggle miễn phí):
- Main dataset (raw): https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset — giải nén sao cho `data/raw/` chứa 2 thư mục `Training/` và `Testing/`.
- Hoặc dùng sẵn file `data.zip` ở repo gốc (~800 MB): giải nén vào root project, kết quả tương đương `data/raw` đầy đủ.

**Cách C — đã copy sẵn `data/processed/*.csv` từ máy khác:** bỏ qua bước này.

Sau khi có `data/raw`, chạy cleaning + split trên máy target:
```bash
python scripts/run_cleaning.py       # loại ảnh hỏng/trùng -> data/processed/clean.csv
python scripts/run_split.py          # stratified 70/15/15 -> train/val/test.csv
python scripts/run_eda.py            # (tuỳ chọn) hình EDA -> results/figures
python scripts/verify_pipeline.py    # kiểm tra pipeline
```

**BraTS 2020** (bài toán grading HGG/LGG, dùng `configs/config_brats.yaml`):
https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation — giải nén rồi trích xuất slice 2D:
```bash
python scripts/prepare_brats.py --src <đường_dẫn_BraTS2020_TrainingData>
# -> PNG slices: data/external_raw/brats2020/{hgg,lgg}/<subject>/sNNN.png
# -> manifests : data/external/brats2020/{train,val,test,all}.csv (subject-level split, 25 slice/patient)
```

### 3. torchvision ImageNet weights (bắt buộc cho lần chạy `train.py` đầu tiên)

Các backbone pretrained sẽ tự download khi có mạng; với máy offline/Restricted thì tải trước và đặt vào cache của torch:

- Linux: `~/.cache/torch/hub/checkpoints/`
- Windows: `%USERPROFILE%\.cache\torch\hub\checkpoints\`

| File | URL | Used by |
|---|---|---|
| `resnet18-f37072fd.pth` | https://download.pytorch.org/models/resnet18-f37072fd.pth | resnet18, mpac_resnet |
| `resnet50-11ad3fa6.pth` | https://download.pytorch.org/models/resnet50-11ad3fa6.pth | resnet50 |
| `vgg16-397923af.pth` | https://download.pytorch.org/models/vgg16-397923af.pth | vgg16 |
| `mobilenet_v2-7ebf99e0.pth` | https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth | mobilenetv2 |
| `efficientnet_b0_rwightman-7f5810bc.pth` | https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth | efficientnetb0 |

> Copy từ máy khác qua USB cũng được — ví dụ thư mục cache có sẵn ở `C:\Users\<user>\.cache\torch\hub\checkpoints`.

### 4. Train (GPU tự detect, chỉnh `training.num_workers` trong `configs/config.yaml` lên 4-8)
```bash
# 1 model
python scripts/train.py --model resnet50 --epochs 30 --batch-size 32

# tất cả 5 models
python scripts/train_all.py --epochs 30 --batch-size 32

# full fine-tune backbone / resume từ last.pt
python scripts/train.py --model resnet50 --finetune
python scripts/train.py --model resnet50 --resume

# test nhanh pipeline (CPU, vài batch)
python scripts/train.py --model custom_cnn --epochs 1 --debug-batches 5
```
Checkpoints: `models/<model>/{best.pt,last.pt,history.csv}` — TensorBoard: `tensorboard --logdir results/logs`.

### 5. Hyperparameter tuning (top models)
```bash
python scripts/tune.py --model resnet50 --trials 8 --epochs 5
```

### 6. Đánh giá & so sánh — 1 lệnh cho tất cả models
```bash
python scripts/evaluate.py --model all          # eval mọi model có best.pt
python scripts/evaluate.py --model resnet50,mobilenetv2   # hoặc subset
python scripts/compare_models.py                # bảng so sánh, McNemar, khuyến nghị
```

### 7. Chạy toàn bộ pipeline trong background (train all -> eval all -> compare)
```bash
# Linux/macOS
nohup python scripts/run_all.py --epochs 30 --batch-size 32 > results/logs/run_all.out 2>&1 &
tail -f results/logs/run_all.out

# Windows PowerShell
Start-Process -NoNewWindow -FilePath .venv\Scripts\python.exe `
  -ArgumentList "scripts\run_all.py --epochs 30 --batch-size 32" `
  -RedirectStandardOutput results\logs\run_all.out -RedirectStandardError results\logs\run_all.err

# tuỳ chọn: chỉ eval + compare (khi đã train xong)
python scripts/run_all.py --skip-train
```
Kết quả: `results/metrics/` (bảng so sánh, predictions, complexity, mcnemar, final_recommendation) và `results/figures/` (confusion matrices, ROC, training curves, Grad-CAM, error analysis).
