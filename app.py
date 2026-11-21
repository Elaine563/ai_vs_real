"""Catch the Rare Fake: a live 'teachable machine' for class imbalance.

1. Pick a few AI-generated images (the rare minority class).
2. Record real photos with your webcam: the app collects ratio x (number of AI images).
3. Train SVM, Random Forest, Small CNN and ResNet18 twice: with no handling and with class weights.
4. See the accuracy paradox happen, then test the models live.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
import threading
import time
import zipfile
from pathlib import Path

import altair as alt
import av
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import live_core as core

st.set_page_config(page_title="Catch the Rare Fake", page_icon="🔍", layout="wide")

GALLERY_DIR = Path(__file__).parent / "ai_gallery"
TEST_REAL = 20          # extra webcam frames held back as the real half of the test set
TEST_AI_MAX = 20        # unpicked gallery images used as the AI half of the test set
FPS = 5                 # burst-mode capture rate

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.big {font-size: 2.1rem; font-weight: 700; line-height: 1.1;}
.muted {color: #7a6a62; font-size: 0.9rem;}
.card {background:#f3ebe2; border-radius:12px; padding:14px 16px; margin-bottom:8px;}
.verdict-ai {color:#b0673a; font-weight:700;}
.verdict-real {color:#3f6b4f; font-weight:700;}
.cm td {border:1px solid #d9cbbf; padding:4px 10px; text-align:center; font-size:0.85rem;}
.cm th {font-size:0.75rem; color:#7a6a62; padding:2px 6px; font-weight:500;}
</style>""", unsafe_allow_html=True)


# ============================================================================ state
class FrameCollector:
    """Receives webcam frames on the WebRTC thread and keeps ~5 per second while recording."""

    def __init__(self):
        self.lock = threading.Lock()
        self.frames: list[np.ndarray] = []
        self.recording = False
        self.target = 0
        self._last = 0.0

    def __call__(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Kept as light as possible: the frame is passed straight back to the browser untouched,
        # and only converted to an image (5 times a second) while recording.
        if not self.recording:
            return frame
        now = time.time()
        if now - self._last >= 1 / FPS:
            img = core.square_resize(frame.to_ndarray(format="rgb24"))
            with self.lock:
                if self.recording and len(self.frames) < self.target:
                    self.frames.append(img)
                    self._last = now
                if len(self.frames) >= self.target:
                    self.recording = False
        return frame

    def snapshot(self):
        with self.lock:
            return list(self.frames), self.recording


ss = st.session_state
ss.setdefault("collector", FrameCollector())
ss.setdefault("picked", [])           # gallery file names
ss.setdefault("uploaded_ai", {})      # name -> uint8 image
ss.setdefault("extra_real", [])       # single shots / uploads
ss.setdefault("run", None)            # results of the last training run


@st.cache_data(show_spinner="Loading the AI image gallery…")
def load_gallery(folder: str) -> dict[str, np.ndarray]:
    files = sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    return {p.name: core.square_resize(Image.open(p)) for p in files[:400]}


@st.cache_resource(show_spinner="Downloading ResNet18 ImageNet weights (first run only)…")
def resnet_backbone():
    return core.load_resnet18_backbone()


gallery = load_gallery(str(GALLERY_DIR))


def ai_train_images() -> list[np.ndarray]:
    return [gallery[n] for n in ss.picked if n in gallery] + list(ss.uploaded_ai.values())


def real_images() -> list[np.ndarray]:
    frames, _ = ss.collector.snapshot()
    return frames + ss.extra_real


# ============================================================================ sidebar (burger menu)
with st.sidebar:
    st.markdown("### 🔍 Catch the Rare Fake")
    page = st.radio("Go to", ["How it works", "1 · Pick AI images", "2 · Record real photos",
                              "3 · Train & compare", "4 · Test it live"], label_visibility="collapsed")
    st.divider()
    ratio = st.select_slider("Imbalance ratio (real : AI)", options=[1, 5, 10, 20, 50], value=10,
                             format_func=lambda r: f"{r}:1")
    n_ai = len(ai_train_images())
    target_train = ratio * n_ai
    st.markdown(f"<div class='card'><span class='muted'>AI images picked</span><div class='big'>{n_ai}</div>"
                f"<span class='muted'>Real photos to record</span><div class='big'>{target_train} + {TEST_REAL}</div>"
                f"<span class='muted'>{ratio} × {n_ai} for training, {TEST_REAL} kept for testing</span></div>",
                unsafe_allow_html=True)
    models = st.multiselect("Models", list(core.MODEL_INFO), default=list(core.MODEL_INFO),
                            format_func=lambda k: core.MODEL_INFO[k][0])
    threshold = st.slider("Decision threshold (AI if score ≥)", 0.05, 0.95, 0.5, 0.05)
    st.divider()
    if st.button("Start over"):
        for k in ("picked", "uploaded_ai", "extra_real", "run"):
            ss.pop(k, None)
        ss.collector = FrameCollector()
        st.rerun()


# ============================================================================ pages
def page_intro():
    st.title("Catch the Rare Fake")
    st.markdown("A live version of the XBDS3024N class-imbalance experiment. You decide how rare the "
                "AI-generated images are, record the real photos yourself, and watch four models learn.")
    c = st.columns(4)
    steps = [("1", "Pick AI images", "A handful of AI-generated images become the rare **minority** class."),
             ("2", "Record real photos", f"Your webcam records **ratio × AI images** real photos in burst mode "
                                         f"({FPS} per second)."),
             ("3", "Train & compare", "SVM, Random Forest, Small CNN and ResNet18 train twice: **no handling** "
                                      "vs **class weights**."),
             ("4", "Test it live", "Show the models a new photo or AI image and see how each one votes.")]
    for col, (n, t, d) in zip(c, steps):
        col.markdown(f"<div class='card'><div class='big'>{n}</div><b>{t}</b><br>{d}</div>", unsafe_allow_html=True)
    st.subheader("What to look for")
    st.markdown(
        "- At **50:1**, the no-handling models often call *everything* real. Their accuracy still looks great, "
        "because almost everything **is** real. That is the accuracy paradox.\n"
        "- **AI recall** (how many AI images were caught) tells the truth. Class weights usually bring it back.\n"
        "- Webcam photos of your room look very different from AI art, which makes the task easier than in the report. To make it harder (and more realistic), also record **paintings, posters, phone screens or printed photos**.\n"
        "- ResNet18 starts with knowledge from 1.2 million ImageNet photos, so it needs far fewer AI examples "
        "than the Small CNN, which learns from scratch.")
    if not gallery:
        st.warning(f"The AI gallery is empty. Run `python prepare_gallery.py path/to/ipcv_subset.zip` "
                   f"(see README) or upload AI images on page 1.")


def page_pick():
    st.header("1 · Pick the AI images (minority class)")
    st.caption(f"{len(gallery)} images in the gallery. Images you don't pick are kept aside to test the models.")
    c1, c2, c3 = st.columns([1, 1, 3])
    n_rand = c1.number_input("How many", 1, 40, 5, label_visibility="collapsed")
    if c2.button(f"Pick {n_rand} at random"):
        rng = np.random.default_rng(int(time.time()))
        ss.picked = list(rng.choice(list(gallery), size=min(n_rand, len(gallery)), replace=False))
        for name in gallery:
            ss[f"pick_{name}"] = name in ss.picked
    if c3.button("Clear picks"):
        ss.picked = []
        for name in gallery:
            ss[f"pick_{name}"] = False

    up = st.file_uploader("Or add your own AI-generated images", type=["png", "jpg", "jpeg", "webp"],
                          accept_multiple_files=True)
    for f in up or []:
        ss.uploaded_ai.setdefault(f.name, core.square_resize(Image.open(f)))

    picked = []
    names = list(gallery)
    for row in range(0, len(names), 8):
        cols = st.columns(8)
        for col, name in zip(cols, names[row:row + 8]):
            col.image(gallery[name], width="stretch")
            if f"pick_{name}" not in ss:
                ss[f"pick_{name}"] = name in ss.picked
            if col.checkbox("use", key=f"pick_{name}", label_visibility="collapsed"):
                picked.append(name)
    if gallery:
        ss.picked = picked
    if ss.uploaded_ai:
        st.markdown(f"**Your uploads ({len(ss.uploaded_ai)})**")
        st.image(list(ss.uploaded_ai.values()), width=96)


def page_record():
    st.header("2 · Record real photos (majority class)")
    col = ss.collector
    need = target_train + TEST_REAL
    if n_ai == 0:
        st.info("Pick at least one AI image first. The number of real photos depends on it.")
    st.markdown(f"Target: **{ratio} × {n_ai} = {target_train}** training photos **+ {TEST_REAL}** test photos "
                f"= **{need}**. Move around, change angle, light and background so the photos aren't all the same.")

    from streamlit_webrtc import WebRtcMode, webrtc_streamer
    left, right = st.columns([3, 2])
    with left:
        webrtc_streamer(key="cam", mode=WebRtcMode.SENDRECV, video_frame_callback=col,
                        media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480},
                                                            "frameRate": {"ideal": 15, "max": 20}},
                                                  "audio": False},
                        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                        async_processing=True)
    with right:
        frames, rec = col.snapshot()
        b1, b2, b3 = st.columns(3)
        if b1.button("● Record", type="primary", disabled=rec or need == 0 or len(frames) >= need):
            with col.lock:
                col.target, col.recording = need, True
            st.rerun()
        if b2.button("■ Stop", disabled=not rec):
            with col.lock:
                col.recording = False
            st.rerun()
        if b3.button("Clear"):
            with col.lock:
                col.frames, col.recording = [], False
            ss.extra_real = []
            st.rerun()
        live_progress(need)

    with st.expander("No webcam stream? Add photos one at a time or upload them"):
        shot = st.camera_input("Single shot")
        if shot is not None and ss.get("last_shot") != shot.file_id:
            ss.last_shot = shot.file_id
            ss.extra_real.append(core.square_resize(Image.open(shot)))
        ups = st.file_uploader("Upload real photos", type=["png", "jpg", "jpeg"], accept_multiple_files=True,
                               key="real_up")
        for f in ups or []:
            key = f"realup_{f.file_id}"
            if key not in ss:
                ss[key] = True
                ss.extra_real.append(core.square_resize(Image.open(f)))

    imgs = real_images()
    if imgs:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for i, im in enumerate(imgs):
                b = io.BytesIO()
                Image.fromarray(im).save(b, "PNG")
                z.writestr(f"real/real_{i:04d}.png", b.getvalue())
        st.download_button(f"Download the {len(imgs)} real photos (.zip)", buf.getvalue(), "real_photos.zip")


def live_progress(need: int):
    frames, rec = ss.collector.snapshot()

    @st.fragment(run_every=1.0 if rec else None)
    def _p():
        f, r = ss.collector.snapshot()
        n = len(f) + len(ss.extra_real)
        st.progress(min(n / max(need, 1), 1.0), text=f"{'🔴 Recording… ' if r else ''}{n} / {need} real photos")
        if f:
            st.image(f[-6:][::-1], width=56)
        if not r and rec:
            st.rerun()          # recording just finished: refresh the buttons

    _p()


def split_data():
    """Training: picked AI + the first ratio x n_ai real frames. Test: unpicked gallery AI + last real frames."""
    ai_tr = ai_train_images()
    reals = real_images()
    n_test_real = min(TEST_REAL, max(len(reals) // 5, 1))
    real_te, real_tr = reals[-n_test_real:], reals[:-n_test_real][:target_train]
    rest = [n for n in gallery if n not in ss.picked]
    rng = np.random.default_rng(42)
    test_names = list(rng.choice(rest, size=min(TEST_AI_MAX, len(rest)), replace=False)) if rest else []
    ai_te = [gallery[n] for n in test_names]
    Xtr = np.stack(real_tr + ai_tr)
    ytr = np.array([0] * len(real_tr) + [1] * len(ai_tr))
    Xte = np.stack(real_te + ai_te) if ai_te else np.stack(real_te)
    yte = np.array([0] * len(real_te) + [1] * len(ai_te))
    return Xtr, ytr, Xte, yte


def page_train():
    st.header("3 · Train & compare")
    reals = real_images()
    problems = []
    if n_ai < 2:
        problems.append("pick at least 2 AI images")
    if len(reals) < 10:
        problems.append("record at least 10 real photos")
    if len(gallery) - len([n for n in ss.picked if n in gallery]) < 3:
        problems.append("keep at least 3 gallery images unpicked so there is something to test on")
    if problems:
        st.warning("Before training: " + "; ".join(problems) + ".")
        return
    if len(reals) < target_train + TEST_REAL:
        st.info(f"You have {len(reals)} of {target_train + TEST_REAL} real photos, so the real training ratio "
                f"will be lower than {ratio}:1. You can still train.")

    if st.button("Train all models", type="primary", disabled=not models):
        Xtr, ytr, Xte, yte = split_data()
        trained, t_all = [], time.time()
        with st.status("Training…", expanded=True) as status:
            if {"svm", "rf"} & set(models):
                st.write("Extracting 1,846 hand-crafted features per image…")
                ext = core.FeatureExtractor()
                Ftr = ext(Xtr)
                for key in [m for m in ("svm", "rf") if m in models]:
                    for fix in (False, True):
                        st.write(f"{core.MODEL_INFO[key][0]} · {'class weights' if fix else 'no handling'}")
                        trained.append(core.train_classical(key, Ftr, ytr, fix, ext))
            bb = None
            if "resnet18" in models:
                try:
                    bb = resnet_backbone()
                except Exception:
                    st.write("ResNet18 skipped: weights could not be downloaded.")
            if bb is not None:
                st.write("ResNet18: turning images into 512 ImageNet features…")
                Etr = core.resnet_embed(bb, Xtr)
                for fix in (False, True):
                    st.write(f"ResNet18 · {'weighted loss' if fix else 'no handling'}")
                    trained.append(core.train_resnet_head(Etr, ytr, fix, bb))
            if "smallcnn" in models:
                for fix in (False, True):
                    bar = st.progress(0.0, text=f"Small CNN · {'weighted loss' if fix else 'no handling'}")
                    trained.append(core.train_smallcnn(Xtr, ytr, fix, progress=lambda p, b=bar: b.progress(p)))
            st.write("Scoring the test images…")
            scores = {(t.key, t.fix): t.predict(Xte) for t in trained}
            status.update(label=f"Done in {time.time() - t_all:.0f} s", state="complete", expanded=False)
        ss.run = {"trained": {(t.key, t.fix): t for t in trained}, "scores": scores, "yte": yte,
                  "n_tr": (int((ytr == 0).sum()), int((ytr == 1).sum())), "ratio_used": ratio,
                  "Xte": Xte, "resnet_missing": "resnet18" in models and bb is None}

    if ss.run:
        show_results(ss.run)


def show_results(run):
    if run.get("resnet_missing"):
        st.warning("ResNet18 was skipped because its ImageNet weights could not be downloaded. Connect to the "
                   "internet and press Train again, or put `resnet18-f37072fd.pth` in the app folder (see README).")
    yte, (nr, na) = run["yte"], run["n_tr"]
    r_used = nr / max(na, 1)
    st.markdown(f"Trained on **{nr} real : {na} AI** (≈ {r_used:.0f}:1). Tested on **{(yte == 0).sum()} new real "
                f"photos** and **{(yte == 1).sum()} AI images the models never saw**.")
    rows = []
    for (key, fix), s in run["scores"].items():
        m = core.evaluate(yte, s, threshold)
        # accuracy if the test set had the same real:AI mix as training (re-weighted by class)
        acc_mix = (m["cm"][0, 0] / max(m["cm"][0].sum(), 1) * r_used + m["ai_recall"]) / (r_used + 1)
        rows.append({"Model": core.MODEL_INFO[key][0], "key": key, "Setting": "Class weights" if fix else "No handling",
                     f"Accuracy at {r_used:.0f}:1": acc_mix, "AI recall": m["ai_recall"],
                     "AI precision": m["ai_precision"], "Macro-F1": m["macro_f1"], "MCC": m["mcc"], "cm": m["cm"]})
    df = pd.DataFrame(rows)

    paradox = df[(df["Setting"] == "No handling") & (df["AI recall"] == 0)]
    if len(paradox):
        names = ", ".join(paradox["Model"])
        st.error(f"**Accuracy paradox:** {names} caught **0** AI images, yet would still score "
                 f"{paradox.iloc[0][f'Accuracy at {r_used:.0f}:1']:.0%} accuracy on a {r_used:.0f}:1 mix. "
                 "They learned to always say *real*.")

    order = [core.MODEL_INFO[k][0] for k in core.MODEL_INFO]
    df["_o"] = df["Model"].map(order.index) * 2 + (df["Setting"] != "No handling")
    df = df.sort_values("_o").drop(columns="_o")
    acc_col = f"Accuracy at {r_used:.0f}:1"
    long = df.melt(id_vars=["Model", "Setting"], value_vars=[f"Accuracy at {r_used:.0f}:1", "AI recall"],
                   var_name="Metric", value_name="Score")
    chart = alt.Chart(long).mark_bar().encode(
        x=alt.X("Setting:N", title=None, sort=["No handling", "Class weights"], axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("Metric:N", sort=[acc_col, "AI recall"]),
        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color("Metric:N", scale=alt.Scale(domain=[acc_col, "AI recall"], range=["#c9b8aa", "#b0673a"]),
                        legend=alt.Legend(orient="top", title=None)),
        column=alt.Column("Model:N", title=None, sort=order),
        tooltip=["Model", "Setting", "Metric", alt.Tooltip("Score:Q", format=".1%")],
    ).properties(width=170, height=220)
    st.altair_chart(chart)

    show = df.drop(columns=["key", "cm"])
    st.dataframe(show.style.format({c: "{:.2f}" for c in show.columns if c not in ("Model", "Setting")}),
                 hide_index=True)

    st.markdown("**Confusion matrices** (rows = actual, columns = predicted)")
    cols = st.columns(4)
    for i, key in enumerate(k for k in core.MODEL_INFO if (k, False) in run["scores"]):
        with cols[i % 4]:
            st.markdown(f"**{core.MODEL_INFO[key][0]}**")
            for setting in ("No handling", "Class weights"):
                cm = df[(df["key"] == key) & (df["Setting"] == setting)]["cm"].iloc[0]
                st.markdown(
                    f"<span class='muted'>{setting}</span><table class='cm'><tr><th></th><th>real</th><th>AI</th></tr>"
                    f"<tr><th>real</th><td>{cm[0, 0]}</td><td>{cm[0, 1]}</td></tr>"
                    f"<tr><th>AI</th><td>{cm[1, 0]}</td><td><b>{cm[1, 1]}</b></td></tr></table>",
                    unsafe_allow_html=True)
    st.caption("Accuracy is re-weighted to your training mix so you can see what it would look like in the real "
               "world, where AI images are just as rare. Webcam frames are similar to each other, so treat these "
               "numbers as a demo, not a proper experiment like the report.")


def page_test():
    st.header("4 · Test it live")
    if not ss.run:
        st.info("Train the models on page 3 first.")
        return
    src = st.radio("Image source", ["Webcam snapshot", "Upload", "A test image"], horizontal=True)
    img = None
    if src == "Webcam snapshot":
        shot = st.camera_input("Take a photo")
        img = core.square_resize(Image.open(shot)) if shot else None
    elif src == "Upload":
        f = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"])
        img = core.square_resize(Image.open(f)) if f else None
    else:
        Xte, yte = ss.run["Xte"], ss.run["yte"]
        i = st.slider("Test image", 0, len(Xte) - 1, int((yte == 0).sum()))
        img = Xte[i]
        st.caption(f"True label: **{'AI-generated' if yte[i] else 'real'}**")
    if img is None:
        return
    left, right = st.columns([1, 3])
    left.image(img, width="stretch")
    with right:
        for key in core.MODEL_INFO:
            if (key, False) not in ss.run["trained"]:
                continue
            c = st.columns([1.2, 2, 2])
            c[0].markdown(f"**{core.MODEL_INFO[key][0]}**")
            for j, fix in enumerate((False, True)):
                s = float(ss.run["trained"][(key, fix)].predict(img[None])[0])
                v = "AI" if s >= threshold else "REAL"
                c[j + 1].markdown(f"<span class='muted'>{'class weights' if fix else 'no handling'}</span> · "
                                  f"<span class='verdict-{v.lower()}'>{v}</span> {s:.2f}", unsafe_allow_html=True)
                c[j + 1].progress(s)


{"How it works": page_intro, "1 · Pick AI images": page_pick, "2 · Record real photos": page_record,
 "3 · Train & compare": page_train, "4 · Test it live": page_test}[page]()
