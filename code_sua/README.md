# FL-LSTM IDS — Federated Learning + LSTM cho Phát hiện Xâm nhập Mạng IoT

## Mô tả
Hệ thống Phát hiện Xâm nhập (IDS) phi tập trung cho mạng Wireless Sensor Network (WSN),
sử dụng Federated Learning kết hợp LSTM và thuật toán FedProx Adaptive-μ.

## Cấu trúc thư mục

```
code_sua/
├── src/                  # Module cốt lõi (Model, Training, Evaluation, FL)
├── experiments/          # Các script chạy thí nghiệm
├── visualization/        # Script vẽ biểu đồ
├── demo/                 # Demo bảo vệ luận văn
├── results/              # Kết quả (ảnh, số liệu, model weights)
│   ├── figures/
│   ├── metrics/
│   └── models/
├── legacy/               # Code Keras/TF cũ (tham khảo)
├── requirements.txt
└── README.md
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy thí nghiệm chính (Adaptive-μ FedProx)

```bash
cd code_sua
python -m experiments.run_adaptive_fedprox
```

## Demo bảo vệ luận văn

```bash
cd code_sua
python -m demo.demo_prototype_ids
```

## Tác giả
- Học viên: [Tên sinh viên]
- GVHD: PGS.TS. Nguyễn Thị Mỹ Bình
- Đơn vị: Trường Đại học Công nghiệp Hà Nội — Khoa CNTT — 2025

## Tài liệu tham khảo chính
- Anwar, S. et al. (2025). Federated Learning-based Intrusion Detection in IoT-WSN. PeerJ CS 11:e2751.
