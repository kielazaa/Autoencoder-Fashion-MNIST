# Autoencoder Fashion-MNIST

## Identitas

- Nama : Zeeda Akiela Ahmad
- NIM : 452024618105

---

## Environment

- Python 3.13.3
- PyTorch
- Torchvision
- NumPy
- Matplotlib

---

## Dataset

Dataset yang digunakan adalah Fashion-MNIST yang terdiri dari:

- 60.000 data training
- 10.000 data testing
- Ukuran gambar 28 × 28 piksel (grayscale)

---

## Training Model

Training dilakukan menggunakan Kaggle Notebook dengan tiga percobaan latent dimension:

- Latent Dimension = 2
- Latent Dimension = 8
- Latent Dimension = 32

Model disimpan dalam format `.pth` setelah proses training selesai.

---

## Menjalankan Program Rekonstruksi

Buka terminal pada folder project kemudian jalankan:

```bash
python reconstruct.py --model models/autoencoder_latent_32.pth --index 10
```

atau jika menggunakan launcher Python di Windows:

```bash
py reconstruct.py --model models/autoencoder_latent_32.pth --index 10
```

---

## Output

Program akan menghasilkan file pada folder `outputs`:

- original.png
- reconstructed.png
- comparison.png

---

## Struktur Folder

```text
Autoencoder-FashionMNIST/
│
├── models/
│   └── autoencoder_latent_32.pth
│
├── outputs/
│   ├── original.png
│   ├── reconstructed.png
│   └── comparison.png
│
├── reconstruct.py
├── requirements.txt
├── README.md
└── Autoencoder_FashionMNIST.ipynb
```
