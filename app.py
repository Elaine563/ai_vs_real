"""Label Lens: catching unlabelled AI images on a photo-sharing app.

A story in six chapters built on the XBDS3024N class-imbalance experiment:
  1 The feed            - can people tell which posts are AI?
  2 When the label falls off - metadata labels vs pixel detection
  3 The lazy moderator  - the accuracy paradox at real-world rarity
  4 Train your own detector - live imbalance lab with the webcam
  5 Label the uploads   - the application: best model screens new posts
  6 The complaint       - false flags and missed fakes, and why humans stay in the loop

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

st.set_page_config(page_title="TrueFrame · Real or rendered?", page_icon="🖼️", layout="wide")

GALLERY_DIR = Path(__file__).parent / "ai_gallery"
TEST_REAL = 20          # extra webcam frames held back as the real half of the test set
TEST_AI_MAX = 20        # unpicked gallery images used as the AI half of the test set
FPS = 5                 # burst-mode capture rate

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,600;0,9..144,800;1,9..144,600&family=Grand+Hotel&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stCaption, p, li, label, input, button {font-family: 'Plus Jakarta Sans', sans-serif;}
h1, h2, h3, h4 {font-family: 'Fraunces', serif !important; letter-spacing: -0.01em;}
.block-container {padding-top: 2.4rem; max-width: 1180px;}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #fff 0%, #f6eff8 100%);}
.logo {font-family: 'Grand Hotel', cursive; font-size: 2.3rem; line-height: 1;
  background: linear-gradient(90deg, #f58529, #dd2a7b 55%, #8134af); -webkit-background-clip: text; color: transparent;}
.brand {font-family: 'Fraunces', serif; font-weight: 800; font-size: 1.35rem; color: #1f1a24;}
.grad {background: linear-gradient(90deg, #f58529, #dd2a7b 55%, #8134af);}
.hero {border-radius: 22px; padding: 26px 30px 22px; color: #fff; margin-bottom: 18px;
  background: radial-gradient(circle at 10% 10%, #feda75 0%, transparent 35%),
              linear-gradient(115deg, #f58529 0%, #dd2a7b 45%, #8134af 80%, #515bd4 100%);
  box-shadow: 0 12px 30px rgba(221,42,123,.25);}
.hero .kicker {font-weight: 700; letter-spacing: .18em; font-size: .75rem; text-transform: uppercase; opacity: .9;}
.hero h1 {color: #fff !important; font-size: 2.5rem; margin: 4px 0 6px; padding: 0;}
.hero p {font-size: 1.02rem; opacity: .96; margin: 0; max-width: 820px;}
.stepper {display: flex; gap: 6px; margin: 0 0 14px;}
.stepper div {flex: 1; height: 6px; border-radius: 99px; background: #eadff0;}
.stepper div.done {background: #dd2a7b; opacity: .45;} .stepper div.now {background: #dd2a7b;}
.muted {color: #6f6478; font-size: 0.9rem;}
.big {font-size: 2.1rem; font-weight: 700; line-height: 1.1;}
.card {background: #fff; border: 1px solid #eee3f1; border-radius: 16px; padding: 14px 16px; margin-bottom: 8px;}
.verdict-ai {color: #d6336c; font-weight: 700;} .verdict-real {color: #2f8f5b; font-weight: 700;}
.cm td {border: 1px solid #e6dcea; padding: 4px 10px; text-align: center; font-size: 0.85rem;}
.cm th {font-size: 0.75rem; color: #6f6478; padding: 2px 6px; font-weight: 500;}
/* social post */
.post {background: #fff; border: 1px solid #ececf1; border-radius: 18px; overflow: hidden; margin-bottom: 6px;
  box-shadow: 0 2px 10px rgba(31,26,36,.05); transition: transform .15s ease, box-shadow .15s ease;}
.post:hover {transform: translateY(-2px); box-shadow: 0 10px 24px rgba(31,26,36,.10);}
.post .top {display: flex; align-items: center; gap: 9px; padding: 9px 12px;}
.ring {width: 32px; height: 32px; border-radius: 50%; padding: 2px; flex: none;
  background: conic-gradient(#feda75, #fa7e1e, #d62976, #962fbf, #4f5bd5, #feda75);}
.ring span {display: flex; width: 100%; height: 100%; border-radius: 50%; background: #fff; border: 2px solid #fff;
  align-items: center; justify-content: center; font-weight: 700; font-size: .8rem; color: #8134af;}
.post .who {font-weight: 700; font-size: .86rem;} .post .loc {font-size: .72rem; color: #8b8194;}
.post .dots {margin-left: auto; color: #8b8194; font-weight: 700;}
.post .pic {position: relative;} .post .pic img {width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;}
.badge {position: absolute; left: 10px; bottom: 10px; background: rgba(20,16,24,.72); color: #fff; font-size: .72rem;
  font-weight: 700; padding: 4px 9px; border-radius: 99px; backdrop-filter: blur(4px);}
.post .acts {display: flex; gap: 14px; padding: 8px 12px 2px; font-size: 1.15rem;} .post .acts .save {margin-left: auto;}
.post .likes {font-weight: 700; font-size: .82rem; padding: 0 12px;}
.post .cap {font-size: .84rem; padding: 2px 12px 10px; color: #2b2430;} .post .cap b {margin-right: 4px;}
.post .tags {padding: 0 12px 10px;}
.stories {display: flex; gap: 16px; overflow-x: auto; padding: 6px 2px 14px;}
.story {text-align: center; font-size: .7rem; color: #4a4152; width: 64px; flex: none;}
.story .ring {width: 58px; height: 58px; margin: 0 auto 4px;} .story .ring span {font-size: 1.1rem;}
.tag {display: inline-block; font-size: 0.72rem; font-weight: 700; padding: 3px 9px; border-radius: 999px; margin: 2px 2px 0 0;}
.tag-ai {background: #d6336c; color: #fff;} .tag-ok {background: #e3f4ea; color: #2f8f5b;}
.tag-wrong {background: #fde8ee; color: #b4234f;} .tag-info {background: #efe7fb; color: #6b3fa0;}
/* detective cards */
.det {background: #fff; border-radius: 20px; padding: 18px; border: 1px solid #eee3f1; height: 100%;
  box-shadow: 0 6px 18px rgba(129,52,175,.08);}
.det .emo {font-size: 2.4rem;} .det h3 {margin: 2px 0 0; font-size: 1.35rem;}
.det .alias {color: #8134af; font-weight: 700; font-size: .82rem; letter-spacing: .04em;}
.stat {margin: 8px 0 0; font-size: .78rem; color: #4a4152;}
.bar {height: 8px; border-radius: 99px; background: #f1e9f5; overflow: hidden; margin-top: 3px;}
.bar i {display: block; height: 100%; border-radius: 99px; background: linear-gradient(90deg, #f58529, #dd2a7b, #8134af);}
.pro, .con {font-size: .82rem; margin-top: 10px;} .pro b {color: #2f8f5b;} .con b {color: #b4234f;}
.phone {max-width: 380px; margin: 0 auto; border-radius: 38px; padding: 14px 12px 10px; background: #111;
  box-shadow: 0 20px 40px rgba(0,0,0,.25);}
.phone .screen {background: #fff; border-radius: 28px; overflow: hidden;}
.phone .bar-top {display: flex; align-items: center; justify-content: space-between; padding: 10px 14px 6px;}
.phone .bar-top .logo {font-size: 1.8rem;} .phone .post {border: 0; border-radius: 0; box-shadow: none;}
.phone .nav {display: flex; justify-content: space-around; padding: 8px 0 6px; font-size: 1.1rem; border-top: 1px solid #eee;}
.where {background: #fff; border: 1px dashed #d9c6e4; border-radius: 16px; padding: 14px 16px; height: 100%;}
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


# ============================================================================ sidebar (chapters)
CHAPTERS = ["The feed", "When the label falls off", "The lazy moderator", "Meet the detectives",
            "The rarity dial", "Label the uploads", "The complaint", "Backstage: train your own detector"]
ICONS = ["📱", "🏷️", "😴", "🕵️", "🎚️", "🛡️", "📩", "⭐"]
ss.setdefault("chapter", CHAPTERS[0])


def go(delta: int):
    ss.chapter = CHAPTERS[(CHAPTERS.index(ss.chapter) + delta) % len(CHAPTERS)]


with st.sidebar:
    st.markdown("<div class='brand'>🖼️ TrueFrame</div><div class='muted'>Real or rendered? Read the pixels.</div>"
                "<div style='margin:14px 0 4px' class='muted'>Protecting</div><div class='logo'>Lumi</div>",
                unsafe_allow_html=True)
    st.write("")
    page = st.radio("Chapter", CHAPTERS, key="chapter", label_visibility="collapsed",
                    format_func=lambda c: (f"⭐  Bonus · {c.split(': ')[1].capitalize()}" if c.startswith("Backstage")
                                           else f"{ICONS[CHAPTERS.index(c)]}  {CHAPTERS.index(c) + 1} · {c}"))
    st.divider()
    st.caption("Built on the XBDS3024N class-imbalance experiment. Lumi is fictional; feed posts are held-out "
               "test images from the Kaggle dataset, and usernames and captions are invented.")

# lab settings (set in the lab chapter, read by the lab functions)
ratio = ss.get("ratio", 10)
models = ss.get("models", list(core.MODEL_INFO))
threshold = ss.get("thr_lab", 0.5)
n_ai = len(ai_train_images())
target_train = ratio * n_ai


# ============================================================================ pages
def page_pick():
    st.subheader("a · Pick the AI posts (the rare minority)")
    st.caption(f"{len(gallery)} images in the gallery. Images you don't pick are kept aside to test the models.")
    c1, c2, c3 = st.columns([1, 1, 3])
    n_rand = c1.number_input("How many", 1, 40, 5, label_visibility="collapsed")
    if c2.button(f"Pick {n_rand} at random"):
        rng = np.random.default_rng(int(time.time()))
        ss.picked = list(rng.choice(list(gallery), size=min(n_rand, len(gallery)), replace=False))
        for name in gallery:
            ss[f"pick_{name}"] = name in ss.picked
        st.rerun()
    if c3.button("Clear picks"):
        ss.picked = []
        for name in gallery:
            ss[f"pick_{name}"] = False
        st.rerun()

    up = st.file_uploader("Or add your own AI-generated images", type=["png", "jpg", "jpeg", "webp"],
                          accept_multiple_files=True)
    new = [f for f in up or [] if f.name not in ss.uploaded_ai]
    for f in new:
        ss.uploaded_ai[f.name] = core.square_resize(Image.open(f))
    if new:
        st.rerun()

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
    if gallery and picked != ss.picked:
        ss.picked = picked
        st.rerun()
    if ss.uploaded_ai:
        st.markdown(f"**Your uploads ({len(ss.uploaded_ai)})**")
        st.image(list(ss.uploaded_ai.values()), width=96)


def page_record():
    st.subheader("b · Record your camera roll (the majority)")
    col = ss.collector
    need = target_train + TEST_REAL
    if n_ai == 0:
        st.info("Pick at least one AI post in step a first. The number of real photos depends on it.")
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
    st.subheader("c · Train & compare")
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
    st.subheader("d · Test it live")
    if not ss.run:
        st.info("Train the models in step c first.")
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




# ============================================================================ shared: the best model + social UI
import base64
import json

BEST_DIR = Path(__file__).parent / "best_model"
DATA_DIR = Path(__file__).parent / "data"
PREVALENCES = [0.001, 0.005, 0.01, 0.02, 0.05, 0.091, 0.2]
USERS = ["wanderlust.mei", "kopi_and_canvas", "jiawen.snaps", "arif_outdoors", "studio.lumen", "ah.boy.eats",
         "nadia.paints", "urban_frames_kl", "sunset.chaser", "pixelpoet", "mamak_diaries", "the.slow.lens",
         "bukit_bound", "inkandoil", "rainy.day.rina", "golden.hour.gary", "kampung.kid", "museum.mood"]
PLACES = ["Penang", "Kuala Lumpur", "Cameron Highlands", "Melaka", "Langkawi", "Ipoh", "Sabah", "Kuching",
          "Studio", "Home", "Gallery Row", "Old Town"]
CAPTIONS = ["Weekend views 🌿", "Can't believe this was real life", "New piece finished today 🎨",
            "Golden hour hits different", "Throwback to last trip ✈️", "Morning coffee ritual ☕",
            "Found this little corner of the city", "Oil on canvas, 3 weeks of work", "No filter needed",
            "Dreamy skies tonight", "Street details I love", "Sunday sketch session", "Lost in the moment",
            "Nature always wins", "Experimenting with light"]
MODEL_KEYS = {"svm": "svm", "random_forest": "rf", "small_cnn": "smallcnn", "resnet18": "resnet18"}


@st.cache_resource(show_spinner="Loading TrueFrame's detector…")
def best_model():
    import torch
    meta = json.loads((BEST_DIR / "meta.json").read_text())
    head = torch.nn.Linear(512, 2)
    head.load_state_dict(torch.load(BEST_DIR / "head.pt", map_location="cpu"))
    head.eval()
    test = np.load(BEST_DIR / "test_scores.npz")
    feed = json.loads((BEST_DIR / "feed.json").read_text()) if (BEST_DIR / "feed.json").exists() else []
    return meta, head, test["scores"], test["labels"], feed


@st.cache_data
def results() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "all_results.csv")


@st.cache_data
def feed_image(name: str) -> np.ndarray:
    return np.asarray(Image.open(BEST_DIR / "feed" / name).convert("RGB"))


def b64(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def post_html(img, user, caption, seed=0, badge="", tags="", place=None) -> str:
    likes = 120 + (seed * 7919) % 4800
    place = place or PLACES[seed % len(PLACES)]
    badge_html = f"<div class='badge'>{badge}</div>" if badge else ""
    return (f"<div class='post'><div class='top'><div class='ring'><span>{user[0].upper()}</span></div>"
            f"<div><div class='who'>{user}</div><div class='loc'>{place}</div></div><div class='dots'>···</div></div>"
            f"<div class='pic'><img src='{b64(img)}'/>{badge_html}</div>"
            f"<div class='acts'><span>♡</span><span>💬</span><span>➤</span><span class='save'>🔖</span></div>"
            f"<div class='likes'>{likes:,} likes</div><div class='cap'><b>{user}</b>{caption}</div>"
            f"{'<div class=tags>' + tags + '</div>' if tags else ''}</div>")


def phone(inner: str) -> str:
    return (f"<div class='phone'><div class='screen'><div class='bar-top'><span class='logo'>Lumi</span>"
            f"<span>♡ &nbsp; ➤</span></div>{inner}<div class='nav'>⌂ ⌕ ⊕ ♡ ◯</div></div></div>")


def need_best_model() -> bool:
    if (BEST_DIR / "meta.json").exists():
        return False
    st.info("This chapter uses TrueFrame's detector (the best model from the experiment), which has not been "
            "trained yet. In the `live_app` folder run once (a few minutes on a laptop CPU):\n\n"
            "`python train_best_model.py \"path/to/ipcv_subset.zip\"`")
    return True


def score_images(head, bb, imgs: list[np.ndarray]) -> np.ndarray:
    import torch
    with torch.no_grad():
        return torch.softmax(head(torch.from_numpy(core.resnet_embed(bb, np.stack(imgs)))), 1)[:, 1].numpy()


def chapter(title: str, lede: str):
    i = CHAPTERS.index(title)
    steps = "".join(f"<div class='{'now' if j == i else 'done' if j < i else ''}'></div>" for j in range(len(CHAPTERS)))
    kicker = "⭐ Bonus chapter" if title.startswith("Backstage") else f"{ICONS[i]} Chapter {i + 1} of {len(CHAPTERS) - 1}"
    st.markdown(f"<div class='stepper'>{steps}</div><div class='hero'><div class='kicker'>{kicker}</div>"
                f"<h1>{title}</h1><p>{lede}</p></div>", unsafe_allow_html=True)


def nav_buttons():
    st.write("")
    i = CHAPTERS.index(ss.chapter)
    a, _, b = st.columns([1, 3, 1])
    if i > 0:
        a.button(f"← {CHAPTERS[i - 1]}", on_click=go, args=(-1,), width="stretch")
    if i < len(CHAPTERS) - 1:
        b.button(f"{CHAPTERS[i + 1]} →", on_click=go, args=(1,), type="primary", width="stretch")


def rates(s_test, y_test, thr):
    rec = float(((s_test >= thr) & (y_test == 1)).sum() / max((y_test == 1).sum(), 1))
    fpr = float(((s_test >= thr) & (y_test == 0)).sum() / max((y_test == 0).sum(), 1))
    return rec, fpr


# ============================================================================ 1 · the feed
def ch_feed():
    chapter("The feed", "Millions of photos are posted to Lumi every day. A few are AI-generated and carry <b>no "
            "label</b>: a holiday that never happened, a painting no one painted. Scroll the feed, make your call on "
            "each post, then see how TrueFrame does.")
    st.markdown("<div class='stories'>" + "".join(
        f"<div class='story'><div class='ring'><span>{u[0].upper()}</span></div>{u[:10]}</div>"
        for u in USERS[:9]) + "</div>", unsafe_allow_html=True)
    if need_best_model():
        return
    meta, head, *_, feed = best_model()
    posts = [p for p in feed if p["role"] == "feed"]
    ss.setdefault("feed_round", 0)
    rnd = ss.feed_round
    rng = np.random.default_rng(100 + rnd)
    real = [p for p in posts if p["label"] == 0]
    fake = [p for p in posts if p["label"] == 1]
    chosen = [real[i] for i in rng.choice(len(real), 6, replace=False)] + \
             [fake[i] for i in rng.choice(len(fake), 3, replace=False)]
    chosen = [chosen[i] for i in rng.permutation(len(chosen))]
    revealed = ss.get(f"revealed_{rnd}", False)
    thr = meta["threshold"]
    cols = st.columns(3)
    right_h = right_m = 0
    for i, p in enumerate(chosen):
        user = USERS[(i + 3 * rnd) % len(USERS)]
        cap = CAPTIONS[(i * 2 + rnd) % len(CAPTIONS)]
        gk = f"guess_{rnd}_{i}"
        badge = tags = ""
        if revealed:
            guess = ss.get(f"saved_guesses_{rnd}", {}).get(gk)
            truth = "AI" if p["label"] else "Real"
            flagged = p["score"] >= thr
            right_h += guess == truth
            right_m += flagged == bool(p["label"])
            badge = "✦ AI info" if flagged else ""
            tags = (f"<span class='tag {'tag-ai' if p['label'] else 'tag-ok'}'>Truth: {truth}</span>"
                    f"<span class='tag {'tag-ok' if guess == truth else 'tag-wrong'}'>You: {guess or '—'}</span>"
                    f"<span class='tag {'tag-ok' if flagged == bool(p['label']) else 'tag-wrong'}'>TrueFrame "
                    f"{p['score']:.2f}</span>")
        with cols[i % 3]:
            st.markdown(post_html(feed_image(p["file"]), user, cap, seed=i + 9 * rnd, badge=badge, tags=tags),
                        unsafe_allow_html=True)
            if not revealed:
                st.segmented_control("Your call", ["Real", "AI"], key=gk, label_visibility="collapsed")
            st.write("")
    c1, c2, _ = st.columns([1, 1, 3])
    if not revealed and c1.button("✨ Reveal", type="primary", width="stretch"):
        ss[f"saved_guesses_{rnd}"] = {k: ss.get(k) for k in list(ss.keys()) if str(k).startswith(f"guess_{rnd}_")}
        ss[f"revealed_{rnd}"] = True
        st.rerun()
    if c2.button("🔄 New feed", width="stretch"):
        ss.feed_round += 1
        st.rerun()
    if revealed:
        st.success(f"**You: {right_h}/9 · TrueFrame: {right_m}/9.** Three of these nine posts were AI-generated. On "
                   "the real Lumi far fewer are, which is exactly what makes them so hard to catch.")
    st.caption("3 AI posts in 9 keeps the game playable; real feeds are far less balanced (Chapter 3).")


# ============================================================================ 2 · metadata vs pixels
AI_WORDS = ["stable diffusion", "midjourney", "dall-e", "dall·e", "openai", "firefly", "imagen", "gemini",
            "comfyui", "novelai", "flux", "leonardo", "ideogram", "chatgpt", "sdxl", "automatic1111", "invokeai"]


def metadata_signals(raw: bytes, img: Image.Image) -> list[str]:
    """Layer 1, as platforms do it: look for AI provenance hints in the file, not the pixels."""
    found = []
    if b"c2pa" in raw or b"jumb" in raw.lower():
        found.append("C2PA Content Credentials block in the file")
    if b"trainedAlgorithmicMedia" in raw:
        found.append("IPTC digital-source type: trainedAlgorithmicMedia (AI-generated)")
    for k, v in (img.info or {}).items():
        text = str(v)[:90].replace("\n", " ")
        if k.lower() in {"parameters", "prompt", "workflow", "negative_prompt", "sd-metadata", "dream"}:
            found.append(f"Generator settings in the file ('{k}': {text}…)")
        elif any(w in text.lower() for w in AI_WORDS):
            found.append(f"'{k}' mentions an AI tool: {text}")
    exif = img.getexif()
    for tag, label in ((305, "Software"), (270, "Description"), (315, "Artist")):
        v = str(exif.get(tag, ""))
        if v and any(w in v.lower() for w in AI_WORDS):
            found.append(f"EXIF {label}: {v[:80]}")
    return found


def ch_metadata():
    chapter("When the label falls off", "Platforms label AI content mostly by reading <b>signals hidden in the "
            "file</b>: Content Credentials (C2PA), IPTC tags, or the uploader ticking a box. A <b>screenshot</b> wipes "
            "those signals but keeps every pixel. TrueFrame adds a second layer that reads the pixels.")
    src = st.segmented_control("What do you want to post?", ["📁 An image file", "📸 A live photo (webcam)"],
                               default="📁 An image file", key="ch2_src")
    if src == "📸 A live photo (webcam)":
        f = st.columns([1.1, 1])[0].camera_input("Snap a photo to post on Lumi")
        st.caption("A real photo taken right now: no AI metadata in the file. Does the pixel layer agree it is real?")
    else:
        f = st.file_uploader("Post an image saved straight from an AI tool (or any photo)",
                             type=["png", "jpg", "jpeg", "webp"])
    if not f:
        st.caption("Tip: some AI tools leave provenance metadata in the files they save. Post one, then flip "
                   "*Simulate a screenshot*. Or switch to a live photo and post yourself.")
        return
    raw = f.getvalue()
    img = Image.open(io.BytesIO(raw))
    shot = st.toggle("📸 Simulate a screenshot (strip all metadata, keep the pixels)", key="strip")
    if shot:
        buf = io.BytesIO()
        Image.fromarray(np.asarray(img.convert("RGB"))).save(buf, "PNG")
        raw = buf.getvalue()
        img = Image.open(io.BytesIO(raw))
    arr = core.square_resize(img)
    sig = metadata_signals(raw, img)
    pix = None
    if (BEST_DIR / "meta.json").exists():
        meta, head, *_ = best_model()
        pix = float(score_images(head, resnet_backbone(), [arr])[0])
        pix_flag = pix >= meta["threshold"]
    left, right = st.columns([1, 1.1])
    with left:
        badge = "✦ AI info" if sig else ""
        st.markdown(phone(post_html(arr, "you", " just posted this", seed=4, badge=badge, place="Lumi")),
                    unsafe_allow_html=True)
    with right:
        st.markdown("### Layer 1 · What the file says")
        if sig:
            st.markdown("<span class='tag tag-ai'>AI info applied from metadata</span>", unsafe_allow_html=True)
            for s_ in sig:
                st.caption("• " + s_)
        else:
            st.markdown("<span class='tag tag-ok'>No signal: the post goes out unlabelled</span>",
                        unsafe_allow_html=True)
            st.caption("Screenshots, re-saves and messaging apps all strip these signals.")
        st.markdown("### Layer 2 · What the pixels say")
        if pix is None:
            need_best_model()
        else:
            st.markdown(f"<span class='tag {'tag-ai' if pix_flag else 'tag-ok'}'>TrueFrame: "
                        f"{'AI info' if pix_flag else 'looks real'}</span> score {pix:.2f}", unsafe_allow_html=True)
            st.progress(pix)
            st.caption("Same pixels, same score, with or without the metadata.")


# ============================================================================ 3 · the lazy moderator
def ch_lazy():
    chapter("The lazy moderator", "Meet a moderator bot that marks <b>every</b> post as real. AI posts are rare, so "
            "it is almost always right, and it catches nothing. This is the <b>accuracy paradox</b> at the heart "
            "of the experiment.")
    if need_best_model():
        return
    meta, head, s_test, y_test, _ = best_model()
    prev = st.select_slider("How many Lumi posts are really AI-generated?", PREVALENCES, value=0.01,
                            format_func=lambda p: f"{p:.1%}")
    rec, fpr = rates(s_test, y_test, meta["threshold"])
    fakes = 10_000 * prev
    lazy_acc, tf_acc = 1 - prev, (1 - prev) * (1 - fpr) + prev * rec
    a, b = st.columns(2)
    with a:
        st.markdown("<div class='det'><div class='emo'>😴</div><h3>Lazy bot</h3>"
                    "<div class='alias'>“Everything is real.”</div></div>", unsafe_allow_html=True)
        st.metric("Accuracy", f"{lazy_acc:.1%}")
        st.metric(f"AI posts caught (of {fakes:,.0f} per 10,000)", "0")
    with b:
        st.markdown("<div class='det'><div class='emo'>🖼️</div><h3>TrueFrame</h3>"
                    "<div class='alias'>ResNet18 + class weights + tuned threshold</div></div>",
                    unsafe_allow_html=True)
        st.metric("Accuracy", f"{tf_acc:.1%}", f"{tf_acc - lazy_acc:+.1%} vs lazy bot", delta_color="off")
        st.metric(f"AI posts caught (of {fakes:,.0f} per 10,000)", f"{fakes * rec:,.0f}")
    st.info("When fakes are rare, the lazy bot can even **beat** TrueFrame on accuracy while catching nothing. In "
            "the experiment, three of four models trained at 50:1 behaved exactly like it: 90.9% accuracy, 0 of 200 "
            "AI images caught. So TrueFrame is judged on **recall, precision and MCC**, never accuracy alone.")


# ============================================================================ 4 · meet the detectives
DETECTIVES = {
    "svm": ("📐", "SVM", "The Geometer", "Draws the widest possible line between real and fake in feature space.",
            "Keeps ranking fakes even when they are rare; moving the threshold alone recovered 40% recall at 50:1.",
            "Ignores class weights and duplicated fakes completely: it already separates its training data perfectly.",
            "Threshold tuning or SMOTE"),
    "random_forest": ("🌳", "Random Forest", "The Committee", "300 decision trees vote on every post.",
                      "With SMOTE it was the only method that improved in every repeated run, and it beat the SVM at "
                      "10:1 with over-sampling.",
                      "Without help its trees all vote 'real': recall fell to 1% at 10:1 and its scores collapsed to 0.",
                      "SMOTE or over-sampling"),
    "small_cnn": ("🐣", "Small CNN", "The Rookie", "Learns everything from scratch, straight from the pixels.",
                  "Flexible: with enough examples it learns its own features, no hand-crafting needed.",
                  "Hit hardest by rarity: 85 AI examples were not enough to learn anything useful.",
                  "Weighted loss, and much more data"),
    "resnet18": ("🦉", "ResNet18", "The Veteran", "Arrives having already studied 1.2 million ImageNet photos.",
                 "Best at every ratio. Trained at 50:1 it ranked fakes as well as the SVM trained on balanced data.",
                 "Can take a shortcut on art style: catches stylised fakes, misses photorealistic ones.",
                 "Class-weighted loss + tuned threshold"),
}


def ch_detectives():
    chapter("Meet the detectives", "TrueFrame was chosen after an audition of four very different detectors, each "
            "trained at three levels of rarity. Every number below comes from the experiment's 54 trained models.")
    R = results()
    pr = R.groupby(["model", "ratio"]).pr_auc.max().unstack()
    f1 = R[R.ratio == 10].groupby("model").macro_f1.max()
    cols = st.columns(4)
    for col, key in zip(cols, ["svm", "random_forest", "small_cnn", "resnet18"]):
        emo, name, alias, how, pro, con, fix = DETECTIVES[key]
        kept = pr.loc[key, 50] / pr.loc[key, 1]
        stats = [("Spotting fakes (PR-AUC, balanced)", pr.loc[key, 1]),
                 ("Spotting fakes (PR-AUC, 50:1)", pr.loc[key, 50]),
                 ("Skill kept when fakes get rare", kept),
                 ("Best macro-F1 at 10:1", f1.loc[key])]
        bars = "".join(f"<div class='stat'>{lab} <b style='float:right'>{v:.2f}</b><div class='bar'>"
                       f"<i style='width:{min(v, 1) * 100:.0f}%'></i></div></div>" for lab, v in stats)
        col.markdown(f"<div class='det'><div class='emo'>{emo}</div><h3>{name}</h3><div class='alias'>{alias}</div>"
                     f"<div class='muted' style='margin-top:6px'>{how}</div>{bars}"
                     f"<div class='pro'><b>Superpower.</b> {pro}</div><div class='con'><b>Weakness.</b> {con}</div>"
                     f"<div class='pro'><b>Best fix:</b> {fix}</div></div>", unsafe_allow_html=True)
    st.write("")
    st.markdown("#### How each detective copes as fakes get rarer")
    long = pr.reset_index().melt(id_vars="model", var_name="ratio", value_name="PR-AUC")
    long["Detective"] = long["model"].map(lambda k: DETECTIVES[k][1])
    long["Training ratio"] = long["ratio"].map(lambda r: f"{r}:1")
    base = alt.Chart(long).encode(
        x=alt.X("Training ratio:N", sort=["1:1", "10:1", "50:1"], axis=alt.Axis(labelAngle=0)),
        y=alt.Y("PR-AUC:Q", scale=alt.Scale(domain=[0, 0.8])),
        color=alt.Color("Detective:N", scale=alt.Scale(domain=["SVM", "Random Forest", "Small CNN", "ResNet18"],
                                                         range=["#f58529", "#2f8f5b", "#8b8194", "#d6336c"]),
                        legend=alt.Legend(orient="top", title=None)))
    chance = alt.Chart(pd.DataFrame({"y": [0.091]})).mark_rule(strokeDash=[4, 4], color="#8b8194").encode(y="y:Q")
    st.altair_chart((base.mark_line(strokeWidth=3) + base.mark_circle(size=90) + chance).properties(height=300),
                    width="stretch")
    st.caption("Best PR-AUC per detective across all imbalance techniques; dashed line = random guessing (0.09).")


# ============================================================================ 5 · the rarity dial
FIXES = {"svm": "smote", "random_forest": "smote", "small_cnn": "weighted_loss", "resnet18": "weighted_loss"}
FIX_NAMES = {"smote": "SMOTE", "weighted_loss": "class-weighted loss"}


def ch_dial():
    chapter("The rarity dial", "Every detector outputs a score; the <b>threshold</b> decides when a score becomes an "
            "AI label. Turn the dial to make fakes rarer and watch the best threshold move.")
    R = results()
    key = st.segmented_control("Detective", list(DETECTIVES), default="random_forest",
                               format_func=lambda k: f"{DETECTIVES[k][0]} {DETECTIVES[k][1]}")
    key = key or "random_forest"
    ratio_ = st.select_slider("🎚️ Training rarity (real : AI)", [1, 10, 50], value=10, format_func=lambda r: f"{r}:1")
    fix = FIXES[key]
    is_svm = key == "svm"
    default_thr = 0.0 if is_svm else 0.5

    def row(tech, mode, r):
        return R[(R.model == key) & (R.technique == tech) & (R.threshold_mode == mode) & (R.ratio == r)].iloc[0]

    raw_t, raw_d, fix_t = row("none", "tuned", ratio_), row("none", "default", ratio_), row(fix, "tuned", ratio_)
    m = st.columns(4)
    m[0].metric("Best threshold, no fix", f"{raw_t.threshold:.2f}", f"default is {default_thr:g}", delta_color="off")
    m[1].metric("Fakes caught at the default", f"{raw_d.recall_fake:.0%}")
    m[2].metric("Fakes caught at the best threshold", f"{raw_t.recall_fake:.0%}",
                f"{raw_t.recall_fake - raw_d.recall_fake:+.0%}", delta_color="normal")
    m[3].metric(f"Best threshold with {FIX_NAMES[fix]}", f"{fix_t.threshold:.2f}")

    rows = []
    for r in (1, 10, 50):
        rows.append({"Training ratio": f"{r}:1", "Setting": "No fix", "Best threshold": row("none", "tuned", r).threshold})
        rows.append({"Training ratio": f"{r}:1", "Setting": f"With {FIX_NAMES[fix]}",
                     "Best threshold": row(fix, "tuned", r).threshold})
    df = pd.DataFrame(rows)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Where the best threshold sits")
        enc = alt.Chart(df).encode(
            x=alt.X("Training ratio:N", sort=["1:1", "10:1", "50:1"], axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Best threshold:Q", title="SVM margin" if is_svm else "Score cut-off"),
            color=alt.Color("Setting:N", scale=alt.Scale(range=["#d6336c", "#8134af"]),
                            legend=alt.Legend(orient="top", title=None)))
        rule = alt.Chart(pd.DataFrame({"y": [default_thr]})).mark_rule(strokeDash=[5, 4], color="#8b8194").encode(y="y:Q")
        st.altair_chart((enc.mark_line(strokeWidth=3) + enc.mark_circle(size=100) + rule).properties(height=280),
                        width="stretch")
    with c2:
        st.markdown("#### Fakes caught, no fix")
        bars = pd.DataFrame([{"Training ratio": f"{r}:1", "Threshold": t, "Recall": row("none", mode, r).recall_fake}
                             for r in (1, 10, 50) for t, mode in (("Default", "default"), ("Best", "tuned"))])
        st.altair_chart(alt.Chart(bars).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
            x=alt.X("Training ratio:N", sort=["1:1", "10:1", "50:1"], axis=alt.Axis(labelAngle=0)),
            xOffset="Threshold:N", y=alt.Y("Recall:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Threshold:N", scale=alt.Scale(domain=["Default", "Best"], range=["#e6d6ee", "#d6336c"]),
                            legend=alt.Legend(orient="top", title=None))).properties(height=280),
            width="stretch")
    if ratio_ == 1:
        fix_note = f"At 1:1 the classes are balanced, so {FIX_NAMES[fix]} changes nothing."
    elif fix_t.threshold > raw_t.threshold + 0.05:
        fix_note = (f"With {FIX_NAMES[fix]}, the best cut-off rises to **{fix_t.threshold:.2f}**: the fix pushes the "
                    "scores of fakes back up, so the threshold no longer has to drop as fakes get rarer.")
    else:
        fix_note = (f"With {FIX_NAMES[fix]}, the best cut-off stays near **{fix_t.threshold:.2f}**: this fix barely "
                    "re-scales the scores, so the threshold still has to move.")
    st.info(f"**Why it moves.** With no fix, the {DETECTIVES[key][1]} learns that almost everything is real, so its "
            f"scores for fakes shrink. At {ratio_}:1 the best cut-off is **{raw_t.threshold:.2f}** instead of "
            f"{default_thr:g}; keeping the default catches **{raw_d.recall_fake:.0%}** of fakes, moving it catches "
            f"**{raw_t.recall_fake:.0%}**. {fix_note} **Two levers, one problem:** re-weight the training, or move "
            "the threshold.")
    if is_svm:
        st.caption("The SVM outputs a signed margin, not a probability: 0 is its default cut-off and lower values "
                   "mean 'label more posts as AI'.")
    st.caption("Best thresholds were tuned on validation data. At 50:1 only 15 validation fakes existed, so those "
               "values are noisy.")


# ============================================================================ 6 · lab
def ch_lab():
    global ratio, models, threshold, n_ai, target_train
    chapter("Backstage: train your own detector", "Go behind the scenes and build a mini detector live. Your webcam is <b>your camera roll</b>, the real "
            "posts, and a few gallery images are the <b>AI minority</b>. Choose how rare the fakes are and watch "
            "what class weights change.")
    c1, c2, c3 = st.columns([1.2, 2, 1.2])
    ratio = c1.select_slider("Real : AI", options=[1, 5, 10, 20, 50], value=ss.get("ratio", 10),
                             format_func=lambda r: f"{r}:1")
    models = c2.multiselect("Detectives", list(core.MODEL_INFO), default=ss.get("models", list(core.MODEL_INFO)),
                            format_func=lambda k: core.MODEL_INFO[k][0])
    threshold = c3.slider("AI if score ≥", 0.05, 0.95, ss.get("thr_lab", 0.5), 0.05)
    ss.ratio, ss.models, ss.thr_lab = ratio, models, threshold
    n_ai = len(ai_train_images())
    target_train = ratio * n_ai
    st.caption(f"{n_ai} AI posts picked → record **{target_train} + {TEST_REAL}** real photos. This lab shows the "
               "imbalance effect live; TrueFrame itself was trained on the 5,500-image dataset, not on webcam frames.")
    t1, t2, t3, t4 = st.tabs(["① Pick AI posts", "② Record camera roll", "③ Train & compare", "④ Test it"])
    with t1:
        page_pick()
    with t2:
        page_record()
    with t3:
        page_train()
    with t4:
        page_test()
    if st.button("Reset the lab"):
        for k in ("picked", "uploaded_ai", "extra_real", "run"):
            ss.pop(k, None)
        ss.collector = FrameCollector()
        st.rerun()


# ============================================================================ 7 · label the uploads
QUEUE_TAG = "<span class='tag tag-wrong'>→ moderator queue</span>"


def ch_label():
    chapter("Label the uploads", "TrueFrame on duty. New posts arrive; layer 1 reads the file, layer 2 reads the "
            "pixels with the <b>best configuration from the experiment</b>. Flagged posts get an <b>AI info</b> "
            "label and a moderator review.")
    if need_best_model():
        return
    meta, head, s_test, y_test, _ = best_model()
    bb = resnet_backbone()
    with st.expander("ℹ️ About TrueFrame's detector"):
        st.markdown(f"**{meta['model']}**, trained on the Kaggle subset at {meta['train_ratio']} "
                    f"({meta['n_train'][0]:,} real : {meta['n_train'][1]:,} AI); threshold {meta['threshold']:.2f} "
                    f"tuned on validation data. On the held-out test set (2,000 real + 200 AI) it caught "
                    f"**{meta['test_recall']:.0%}** of AI images with **{meta['test_precision']:.0%}** precision and a "
                    f"**{meta['test_fpr']:.1%}** false-alarm rate (macro-F1 {meta['test_macro_f1']:.3f}, "
                    f"MCC {meta['test_mcc']:.3f}).")
    c1, c2 = st.columns(2)
    thr = c1.slider("Label as AI if score ≥", 0.05, 0.95, float(round(meta["threshold"], 2)), 0.01)
    prev = c2.select_slider("Share of Lumi uploads that are really AI", PREVALENCES, value=0.01,
                            format_func=lambda p: f"{p:.1%}")
    rec, fpr = rates(s_test, y_test, thr)
    n = 10_000
    caught, fa = n * prev * rec, n * (1 - prev) * fpr
    prec = caught / max(caught + fa, 1e-9)
    m = st.columns(4)
    m[0].metric("AI uploads / 10k", f"{n * prev:,.0f}")
    m[1].metric("Labelled correctly", f"{caught:,.0f}", f"{rec:.0%} recall", delta_color="off")
    m[2].metric("Creators wrongly labelled", f"{fa:,.0f}", f"{fpr:.1%} of genuine", delta_color="off")
    m[3].metric("Labels that are right", f"{prec:.0%}")
    grid = np.linspace(0.05, 0.95, 37)
    curve = []
    for t in grid:
        r_, f_ = rates(s_test, y_test, t)
        p_ = r_ * prev / max(r_ * prev + f_ * (1 - prev), 1e-9)
        curve += [{"Threshold": t, "Rate": r_, "Line": "Fakes caught (recall)"},
                  {"Threshold": t, "Rate": f_, "Line": "Genuine posts flagged"},
                  {"Threshold": t, "Rate": p_, "Line": f"Labels right at {prev:.1%} AI"}]
    lines = alt.Chart(pd.DataFrame(curve)).mark_line(strokeWidth=3).encode(
        x="Threshold:Q", y=alt.Y("Rate:Q", axis=alt.Axis(format="%")),
        color=alt.Color("Line:N", scale=alt.Scale(range=["#d6336c", "#8b8194", "#8134af"]),
                        legend=alt.Legend(orient="top", title=None)))
    now = alt.Chart(pd.DataFrame({"x": [thr]})).mark_rule(color="#f58529", strokeWidth=2).encode(x="x:Q")
    st.altair_chart((lines + now).properties(height=240), width="stretch")
    st.caption("Move the threshold: the orange line is your setting. Rates are measured on the report's held-out test set.")
    st.divider()
    files = st.file_uploader("📤 New uploads: drop one or more images", type=["png", "jpg", "jpeg", "webp"],
                             accept_multiple_files=True, key="queue")
    if not files:
        st.caption("Try a mix: photos from your phone, a photo of a painting, and images from an AI tool, "
                   "including a photorealistic one.")
        return
    show_cam = st.toggle("🔥 Show where TrueFrame looked (Grad-CAM)", value=False)
    raws = [f.getvalue() for f in files]
    pil = [Image.open(io.BytesIO(r)) for r in raws]
    imgs = [core.square_resize(p) for p in pil]
    scores = score_images(head, bb, imgs)
    flagged_n = 0
    cols = st.columns(3)
    for i, (f, raw, p, img, s_) in enumerate(zip(files, raws, pil, imgs, scores)):
        sig = metadata_signals(raw, p)
        flag = bool(sig) or s_ >= thr
        flagged_n += flag
        why = "metadata" if sig else "pixels"
        shown = core.overlay(img, core.gradcam(bb, head, img)) if show_cam else img
        tags = (f"<span class='tag {'tag-ai' if flag else 'tag-ok'}'>{'AI info · ' + why if flag else 'No label'}</span>"
                f"<span class='tag tag-info'>pixel score {s_:.2f}</span>"
                + (QUEUE_TAG if flag else ""))
        with cols[i % 3]:
            st.markdown(post_html(shown, USERS[i % len(USERS)], " " + f.name, seed=i + 20,
                                  badge="✦ AI info" if flag else "", tags=tags), unsafe_allow_html=True)
    st.success(f"**Moderation queue:** {flagged_n} of {len(files)} uploads flagged for review.")


# ============================================================================ 8 · the complaint
def ch_complaint():
    chapter("The complaint", "TrueFrame is not perfect, and the experiment says exactly where it fails. These are its "
            "<b>real mistakes</b> on the held-out test set, not staged examples.")
    if need_best_model():
        return
    meta, head, *_, feed = best_model()
    bb = resnet_backbone()
    ff = [p for p in feed if p["role"] == "false_flag"]
    missed = [p for p in feed if p["role"] == "missed"]
    st.markdown("### 📩 “My work got an AI label, but I made it myself”")
    st.markdown("Genuine images TrueFrame labelled as AI. The heatmap shows where it looked: often a painterly "
                "style or smooth texture, the same **style shortcut** the experiment's Grad-CAM analysis found.")
    if not ff:
        st.caption("No genuine test image was labelled at the tuned threshold.")
    cols = st.columns(4)
    for i, p in enumerate(ff):
        img = feed_image(p["file"])
        with cols[i % 4]:
            st.markdown(post_html(core.overlay(img, core.gradcam(bb, head, img)), USERS[(i + 6) % len(USERS)],
                                  " my own work 🙁", seed=i + 40, badge="✦ AI info",
                                  tags=f"<span class='tag tag-wrong'>wrongly labelled · {p['score']:.2f}</span>"),
                        unsafe_allow_html=True)
    st.markdown("### 🕵️ The fakes that slipped through")
    st.markdown("AI images that went out with no label. Most are **photorealistic**, the hardest kind, as the report found.")
    if not missed:
        st.caption("No AI test image was missed at the tuned threshold.")
    cols = st.columns(4)
    for i, p in enumerate(missed):
        img = feed_image(p["file"])
        with cols[i % 4]:
            st.markdown(post_html(core.overlay(img, core.gradcam(bb, head, img)), USERS[(i + 11) % len(USERS)],
                                  " living my best life ✨", seed=i + 60,
                                  tags=f"<span class='tag tag-wrong'>missed · {p['score']:.2f}</span>"),
                        unsafe_allow_html=True)
    st.markdown("### ✅ What Lumi should do")
    st.markdown(
        "1. **Never auto-punish.** A pixel-based flag starts a human review and an appeal: when fakes are rare, many "
        "flags are genuine work.\n"
        "2. **Use both layers.** Trust provenance metadata when present; use the pixel detector when it is missing.\n"
        "3. **Tune the threshold to the team.** Set it from moderator capacity and the expected share of fakes.\n"
        "4. **Retrain on what it misses.** Add photorealistic fakes and genuine paintings to break the style shortcut.")
    st.markdown("### 🌍 Where else could this go?")
    w = st.columns(3)
    for col, (emo, title, need) in zip(w, [
            ("🚗", "Insurance claims", "Retrain on car and property damage photos plus AI-edited damage; the current "
                                       "model has never seen a dent."),
            ("📰", "Newsroom verification", "Retrain on news and event photos plus synthetic 'breaking news' images."),
            ("🪪", "eKYC selfie checks", "Retrain on live selfies and AI-generated faces; add liveness checks.")]):
        col.markdown(f"<div class='where'><div style='font-size:1.8rem'>{emo}</div><b>{title}</b>"
                     f"<div class='muted' style='margin-top:4px'>{need}</div></div>", unsafe_allow_html=True)
    st.caption("A detector transfers only to data that looks like its training data, so each use case needs its own data.")


{"The feed": ch_feed, "When the label falls off": ch_metadata, "The lazy moderator": ch_lazy,
 "Meet the detectives": ch_detectives, "The rarity dial": ch_dial, "Backstage: train your own detector": ch_lab,
 "Label the uploads": ch_label, "The complaint": ch_complaint}[page]()
nav_buttons()
