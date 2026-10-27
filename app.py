from pathlib import Path
from io import BytesIO
import random
import zipfile

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageEnhance

# ==========================================
# CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AstroSurface AI",
    page_icon="🌙",
    layout="wide"
)

IMAGE_SIZE = 128
LATENT_SIZE = 64
NUM_CLASSES = 3
DEVICE = torch.device("cpu")

MODEL_DIR = Path("outputs")          # change to Path("outputs") if needed
DEFAULT_MODEL = "conditional_generator.pth"
HISTORY_LIMIT = 8

SCENARIOS = {
    "Small": {
        "label": 0,
        "icon": "🔹",
        "desc": "Compact bowl-shaped crater with a sharp rim.",
    },
    "Medium": {
        "label": 1,
        "icon": "🔸",
        "desc": "Flat-floored crater with a terraced wall and ejecta.",
    },
    "Large": {
        "label": 2,
        "icon": "🌑",
        "desc": "Wide terraced crater with a central peak and rays.",
    },
}

TINTS = {
    "Grayscale": None,
    "Warm lunar": (1.00, 0.95, 0.85),
    "Cool moonlight": (0.88, 0.95, 1.00),
    "Sepia": (1.00, 0.88, 0.70),
}


# ==========================================
# STYLING
# ==========================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(99, 102, 241, 0.18), transparent 40%),
            radial-gradient(circle at 85% 30%, rgba(236, 72, 153, 0.12), transparent 45%),
            linear-gradient(180deg, #05070f 0%, #0b1020 100%);
    }

    #MainMenu, footer {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        max-width: 1250px;
    }

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(15,23,42,0.6));
        border: 1px solid rgba(148, 163, 184, 0.25);
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        font-size: 2.6rem;
        margin: 0;
        background: linear-gradient(90deg, #e0e7ff, #a5b4fc, #f0abfc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        color: #cbd5e1;
        margin: 0.6rem 0 0 0;
        font-size: 1.05rem;
    }

    .card {
        padding: 1.2rem 1.4rem;
        border-radius: 16px;
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.2);
        margin-bottom: 1rem;
    }
    .card h4 { margin: 0 0 0.3rem 0; color: #e2e8f0; }
    .card p  { margin: 0; color: #94a3b8; font-size: 0.95rem; }

    .badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.8rem;
        background: rgba(99, 102, 241, 0.2);
        color: #c7d2fe;
        border: 1px solid rgba(99, 102, 241, 0.4);
        margin-right: 0.4rem;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #6366f1, #a855f7);
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1rem;
        font-weight: 700;
        color: white;
    }
    div.stButton > button[kind="primary"]:hover { filter: brightness(1.15); }
    div.stButton > button[kind="secondary"] { border-radius: 12px; }

    div.stDownloadButton > button {
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.5);
    }

    [data-testid="stImage"] img {
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.3);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
    }

    [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 14px;
        padding: 0.8rem 1rem;
    }

    .stTabs [data-baseweb="tab"] { font-weight: 500; font-size: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# CONDITIONAL GENERATOR
# Must match the training architecture
# ==========================================

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_embedding = nn.Embedding(NUM_CLASSES, 32)

        self.project = nn.Sequential(
            nn.Linear(LATENT_SIZE + 32, 256 * 4 * 4),
            nn.ReLU(True)
        )

        self.network = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_features = self.label_embedding(labels)
        combined = torch.cat([noise, label_features], dim=1)

        x = self.project(combined)
        x = x.view(-1, 256, 4, 4)

        return self.network(x)


# ==========================================
# MODEL HELPERS
# ==========================================

def list_models():
    """All generator checkpoints in the model folder."""
    files = sorted(MODEL_DIR.glob("conditional_generator*.pth"))
    names = [f.name for f in files]

    if DEFAULT_MODEL in names:
        names.remove(DEFAULT_MODEL)
        names.insert(0, DEFAULT_MODEL)

    return names


@st.cache_resource
def load_generator(model_name):
    path = MODEL_DIR / model_name

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}. Run train_conditional_gan.py first."
        )

    model = Generator().to(DEVICE)

    state_dict = torch.load(
        path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


def generate_raw(generator, label, seed):
    """Generate one raw 128 x 128 grayscale image (PIL, mode L)."""
    torch.manual_seed(int(seed))

    noise = torch.randn(1, LATENT_SIZE, device=DEVICE)
    labels = torch.tensor([label], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        generated = generator(noise, labels)

    pixels = generated[0, 0].cpu().numpy()
    pixels = np.clip((pixels + 1.0) * 127.5, 0, 255).astype(np.uint8)

    return Image.fromarray(pixels, mode="L")


# ==========================================
# IMAGE HELPERS
# ==========================================

def render(raw, size, contrast, sharpness, tint_name):
    """Apply the sidebar display settings to a raw image."""
    image = raw.resize((size, size), Image.Resampling.LANCZOS)

    if contrast != 1.0:
        image = ImageEnhance.Contrast(image).enhance(contrast)

    if sharpness != 1.0:
        image = ImageEnhance.Sharpness(image).enhance(sharpness)

    tint = TINTS[tint_name]
    if tint is not None:
        arr = np.asarray(image, dtype=np.float32)[..., None]
        arr = arr * np.array(tint, dtype=np.float32)
        image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")

    return image


def to_png_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def to_zip_bytes(items):
    """items: list of (filename, PIL image)."""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename, image in items:
            archive.writestr(filename, to_png_bytes(image))
    return buffer.getvalue()


def variety_score(raw_images):
    """Average pixel spread across images. Higher = more variety."""
    stack = np.stack(
        [np.asarray(img, dtype=np.float32) / 255.0 for img in raw_images]
    )
    return float(stack.std(axis=0).mean())


def add_to_history(raw, scenario, seed):
    history = st.session_state["history"]
    history.insert(0, {"raw": raw, "scenario": scenario, "seed": int(seed)})
    del history[HISTORY_LIMIT:]


def randomize_seed():
    st.session_state["seed"] = random.randint(0, 999999)


# ==========================================
# SESSION STATE
# ==========================================

if "seed" not in st.session_state:
    st.session_state["seed"] = 42

if "result" not in st.session_state:
    st.session_state["result"] = None

if "history" not in st.session_state:
    st.session_state["history"] = []


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.markdown("### 🧠 Model")

    available_models = list_models()

    if available_models:
        model_name = st.selectbox(
            "Checkpoint",
            options=available_models,
            index=0,
            help="Earlier checkpoints sometimes give more variety.",
        )
        st.success(f"{len(available_models)} model file(s) found")
    else:
        model_name = DEFAULT_MODEL
        st.warning(
            f"No models found in '{MODEL_DIR}'. Train the model first."
        )

    st.divider()
    st.markdown("### 🎨 Image controls")

    size_choice = st.select_slider(
        "Output size",
        options=[256, 512, 768],
        value=512,
    )
    contrast = st.slider("Contrast", 0.6, 1.8, 1.0, 0.05)
    sharpness = st.slider("Sharpness", 0.0, 3.0, 1.0, 0.1)
    tint_name = st.selectbox("Color tint", list(TINTS.keys()))

    st.caption(
        "These only change how images are displayed and downloaded. "
        "They don't change the model."
    )


def styled(raw):
    return render(raw, size_choice, contrast, sharpness, tint_name)


# ==========================================
# HERO HEADER
# ==========================================

st.markdown(
    """
    <div class="hero">
        <h1>🌙 AstroSurface AI</h1>
        <p>Conditional GAN-based synthetic lunar terrain generation.
        Pick a crater scenario and generate a hypothetical surface image.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Research prototype: outputs are synthetic illustrations, "
    "not physically validated predictions of lunar impacts."
)

tab_generate, tab_compare, tab_variety, tab_batch, tab_about = st.tabs(
    [
        "🚀 Generate",
        "🔍 Compare scenarios",
        "🎲 Variety test",
        "📦 Batch",
        "ℹ️ About",
    ]
)


# ==========================================
# TAB 1: GENERATE
# ==========================================

with tab_generate:
    left, right = st.columns([1, 1.3], gap="large")

    with left:
        st.markdown("### Scenario controls")

        scenario_name = st.radio(
            "Crater scenario",
            options=list(SCENARIOS.keys()),
            format_func=lambda n: f"{SCENARIOS[n]['icon']}  {n}",
            horizontal=True,
        )

        info = SCENARIOS[scenario_name]
        st.markdown(
            f"""
            <div class="card">
                <h4>{info['icon']} {scenario_name} crater</h4>
                <p>{info['desc']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        seed_col, dice_col = st.columns([3, 1])

        with seed_col:
            seed = st.number_input(
                "Random seed",
                min_value=0,
                max_value=999999,
                step=1,
                key="seed",
            )

        with dice_col:
            st.write("")
            st.write("")
            st.button("🎲", on_click=randomize_seed, help="Random seed")

        generate = st.button(
            "🚀 Generate terrain",
            type="primary",
            use_container_width=True,
        )

        st.caption("Same seed + same scenario always gives the same image.")

    with right:
        st.markdown("### Generated terrain")

        if generate:
            try:
                generator = load_generator(model_name)
                raw = generate_raw(
                    generator, SCENARIOS[scenario_name]["label"], seed
                )
                st.session_state["result"] = {
                    "raw": raw,
                    "scenario": scenario_name,
                    "seed": int(seed),
                    "model": model_name,
                }
                add_to_history(raw, scenario_name, seed)
            except FileNotFoundError as error:
                st.error(str(error))
            except Exception as error:
                st.error(f"Generation failed: {error}")

        result = st.session_state["result"]

        if result is not None:
            image = styled(result["raw"])

            st.image(
                image,
                caption=f"Synthetic {result['scenario']} scenario  |  seed {result['seed']}",
                width=min(size_choice, 640),
            )

            st.markdown(
                f"""
                <span class="badge">{result['scenario']}</span>
                <span class="badge">seed {result['seed']}</span>
                <span class="badge">{size_choice} x {size_choice} px</span>
                <span class="badge">{result['model']}</span>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            st.download_button(
                label="⬇️ Download PNG",
                data=to_png_bytes(image),
                file_name=f"astrosurface_{result['scenario'].lower()}_{result['seed']}.png",
                mime="image/png",
                use_container_width=True,
            )
        else:
            st.markdown(
                """
                <div class="card">
                    <h4>No image yet</h4>
                    <p>Choose a scenario and press Generate terrain.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # History strip
    history = st.session_state["history"]

    if history:
        st.divider()
        head_col, clear_col = st.columns([5, 1])

        with head_col:
            st.markdown("### 🕘 Recent images")

        with clear_col:
            if st.button("Clear", key="clear_history"):
                st.session_state["history"] = []
                st.rerun()

        columns = st.columns(4)

        for i, item in enumerate(history):
            with columns[i % 4]:
                st.image(
                    styled(item["raw"]),
                    caption=f"{item['scenario']} | seed {item['seed']}",
                    use_container_width=True,
                )


# ==========================================
# TAB 2: COMPARE
# ==========================================

with tab_compare:
    st.markdown("### Small vs Medium vs Large")
    st.caption("Each scenario uses a different seed so the terrain varies.")

    compare_seed = st.number_input(
        "Comparison seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
        key="compare_seed",
    )

    if st.button("🔍 Compare all three", type="primary"):
        try:
            generator = load_generator(model_name)
            columns = st.columns(3, gap="medium")

            for idx, (column, (name, info)) in enumerate(
                zip(columns, SCENARIOS.items())
            ):
                with column:
                    current_seed = int(compare_seed) + idx
                    raw = generate_raw(generator, info["label"], current_seed)
                    image = styled(raw)

                    st.image(
                        image,
                        caption=f"{info['icon']} {name}  |  seed {current_seed}",
                        use_container_width=True,
                    )
                    st.download_button(
                        label=f"⬇️ {name}",
                        data=to_png_bytes(image),
                        file_name=f"astrosurface_{name.lower()}_{current_seed}.png",
                        mime="image/png",
                        key=f"download_{name}",
                        use_container_width=True,
                    )
        except FileNotFoundError as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"Generation failed: {error}")


# ==========================================
# TAB 3: VARIETY TEST
# ==========================================

with tab_variety:
    st.markdown("### Variety test")
    st.caption(
        "Six different seeds for one scenario. If they all look almost "
        "the same, the model has collapsed: pick another checkpoint in the sidebar."
    )

    variety_scenario = st.radio(
        "Scenario to test",
        options=list(SCENARIOS.keys()),
        format_func=lambda n: f"{SCENARIOS[n]['icon']}  {n}",
        horizontal=True,
        key="variety_scenario",
    )

    if st.button("🎲 Run variety test", type="primary"):
        try:
            generator = load_generator(model_name)
            label = SCENARIOS[variety_scenario]["label"]

            raws = [generate_raw(generator, label, s) for s in range(1, 7)]

            st.metric(
                "Variety score",
                f"{variety_score(raws):.3f}",
                help=(
                    "Average pixel difference between the six images. "
                    "Higher means more variety. Compare scores across "
                    "checkpoints to pick the most varied one."
                ),
            )

            for row_start in (0, 3):
                columns = st.columns(3, gap="medium")
                for offset, column in enumerate(columns):
                    index = row_start + offset
                    with column:
                        st.image(
                            styled(raws[index]),
                            caption=f"seed {index + 1}",
                            use_container_width=True,
                        )
        except FileNotFoundError as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"Generation failed: {error}")


# ==========================================
# TAB 4: BATCH
# ==========================================

with tab_batch:
    st.markdown("### Batch generate")
    st.caption("Create several images at once and download them as a ZIP.")

    batch_cols = st.columns(3)

    with batch_cols[0]:
        batch_scenario = st.selectbox(
            "Scenario",
            options=list(SCENARIOS.keys()),
            key="batch_scenario",
        )

    with batch_cols[1]:
        batch_count = st.slider("How many", 2, 12, 6, key="batch_count")

    with batch_cols[2]:
        batch_start = st.number_input(
            "Starting seed",
            min_value=0,
            max_value=999000,
            value=100,
            step=1,
            key="batch_start",
        )

    if st.button("📦 Generate batch", type="primary"):
        try:
            generator = load_generator(model_name)
            label = SCENARIOS[batch_scenario]["label"]

            items = []
            columns = st.columns(4)

            for i in range(batch_count):
                current_seed = int(batch_start) + i
                raw = generate_raw(generator, label, current_seed)
                image = styled(raw)
                filename = f"astrosurface_{batch_scenario.lower()}_{current_seed}.png"
                items.append((filename, image))

                with columns[i % 4]:
                    st.image(
                        image,
                        caption=f"seed {current_seed}",
                        use_container_width=True,
                    )

            st.download_button(
                label=f"⬇️ Download all {batch_count} as ZIP",
                data=to_zip_bytes(items),
                file_name=f"astrosurface_{batch_scenario.lower()}_batch.zip",
                mime="application/zip",
                use_container_width=True,
            )
        except FileNotFoundError as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"Generation failed: {error}")


# ==========================================
# TAB 5: ABOUT
# ==========================================

with tab_about:
    st.markdown("### How it works")

    st.markdown(
        """
        <div class="card">
            <h4>1. Input</h4>
            <p>A random noise vector (picked by the seed) and the selected scenario label.</p>
        </div>
        <div class="card">
            <h4>2. Generator</h4>
            <p>A conditional GAN generator creates a 128 x 128 grayscale terrain image.</p>
        </div>
        <div class="card">
            <h4>3. Conditional control</h4>
            <p>The scenario label (small, medium, large) steers the crater size and shape.</p>
        </div>
        <div class="card">
            <h4>4. Output</h4>
            <p>The image is upscaled for display, with optional contrast, sharpness and tint.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Dataset: procedurally generated synthetic crater-like images. "
        "These are illustrations, not real lunar observations."
    )