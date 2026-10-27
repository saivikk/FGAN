# 🌙 AstroSurface AI

**Conditional GAN-based generation of hypothetical lunar crater terrain images.**

Pick a crater scenario (Small, Medium or Large), choose a seed, and generate a synthetic lunar surface image. Built with **PyTorch** and **Streamlit**.

> ⚠️ **Research prototype.** All training data is procedurally generated (synthetic). The outputs are illustrations, **not** real lunar imagery and **not** physically validated predictions of lunar impacts.

---

## 📸 Screenshots

<!-- Add your screenshots to a /screenshots folder and update the paths below -->

| Generate | Compare scenarios |
|---|---|
| ![Generate](screenshots/generate.png) | ![Compare](screenshots/compare.png) |

---

## ✨ Features

- **Conditional generation:** the selected scenario (small, medium, large) controls the crater type.
- **Reproducible seeds:** the same seed and scenario always give the same image.
- **Compare scenarios:** see small, medium and large craters side by side.
- **Variety test:** generate six seeds at once and get a variety score to detect mode collapse.
- **Batch export:** generate up to 12 images and download them as a ZIP.
- **Checkpoint selector:** switch between saved model checkpoints from the sidebar.
- **Image controls:** adjust output size, contrast, sharpness and tint (display only).

---

## 🧠 How it works

1. **Data:** `prepare_data.py` builds a synthetic dataset of 1,500 grayscale 128 × 128 crater images (500 per class). Each class has its own morphology:

   | | Small | Medium | Large |
   |---|---|---|---|
   | Shape | Simple bowl | Flat floor, one terrace | Wide, two terraces |
   | Central peak | No | No | Yes |
   | Ejecta and rays | Minimal | Moderate | Strong |

   Images use a height map (bowl, rim, ejecta, peak, fractal terrain), surface-normal lighting, 2× supersampling and light grain.

2. **Model:** a **Conditional DCGAN** trained from scratch in PyTorch.
   - **Generator:** 64-d noise + 32-d label embedding → five transposed-convolution layers → 128 × 128 image (tanh).
   - **Discriminator:** image + label map + minibatch-variety channel → five strided convolutions → real/fake score.

3. **Training:** BCE loss, Adam (lr 2e-4, β₁ = 0.5), batch size 16, 100 epochs, one-sided label smoothing (0.9), decaying instance noise, checkpoints every 5 epochs. Runs on CPU.

4. **App:** `app.py` loads the trained generator and serves it through Streamlit.

---

## 📁 Project structure

```
AstroSurface-AI/
├── prepare_data.py            # Generates the synthetic crater dataset
├── train_conditional_gan.py   # Trains the conditional GAN
├── train_gan.py               # Earlier unconditional DCGAN (baseline)
├── app.py                     # Streamlit web application
├── dataset/                   # Generated images (small/, medium/, large/)
├── outputs/                   # Trained models and training previews
├── screenshots/               # README images
└── README.md
```

---

## ⚙️ Installation

**Requirements:** Python 3.11 (tested on 3.11.9). No GPU needed.

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install torch torchvision numpy matplotlib pillow streamlit
```

---

## 🚀 Usage

Run the steps in order.

**1. Generate the dataset**
```bash
python prepare_data.py
```
Creates `dataset/small`, `dataset/medium` and `dataset/large`.

**2. Train the model**
```bash
python train_conditional_gan.py
```
Saves the generator and checkpoints into `outputs/` and writes a preview image every 5 epochs. Training on CPU takes a while, and checkpoints are saved along the way, so you can stop early.

**3. Launch the app**
```bash
streamlit run app.py
```
Open the local URL that Streamlit prints (usually `http://localhost:8501`).

> Make sure `MODEL_DIR` in `app.py` points to the folder containing your trained model (`outputs` by default).

---

## 🔬 Evaluation

Evaluation is qualitative and diagnostic:

- Fixed-noise preview grids across epochs.
- Whether classes differ in **shape**, not just size.
- Sample diversity, checked visually and with the in-app **variety score**.

Distribution-level metrics such as FID have not been computed yet.

---

## ⚠️ Limitations

- Trained on **synthetic** data, so the model learns the structure of the procedural generator and not real lunar geology.
- Not a physics simulation and not a scientific prediction of impacts.
- Low resolution (128 × 128), chosen for CPU-feasible training.
- GAN training can be unstable. Mode collapse and checkpoint-to-checkpoint variation are possible, and the checkpoint selector exists for that reason.

---

## 🔭 Future work

- Train on real lunar imagery from public archives.
- Higher resolution and stronger generators (StyleGAN, diffusion models).
- Extra controls such as crater depth, impact angle and terrain type.
- Quantitative evaluation with FID and SSIM.
- Public deployment (for example Streamlit Community Cloud).

---

## 🛠️ Tech stack

Python · PyTorch · Streamlit · NumPy · Pillow · Matplotlib

---

## 👤 Author

**[Your Name]**
[Institution] · [Email or LinkedIn]
Mentor: [Mentor Name]

---

## 📄 License

[Choose a license, e.g. MIT] — see `LICENSE`.

---

## 🙏 Acknowledgments

- Goodfellow et al., *Generative Adversarial Nets* (2014)
- Mirza & Osindero, *Conditional Generative Adversarial Nets* (2014)
- Radford et al., *DCGAN* (2016)
