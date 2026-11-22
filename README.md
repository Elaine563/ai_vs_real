# TrueFrame · Real or rendered?

A story-driven Streamlit application built on the XBDS3024N class-imbalance experiment. **TrueFrame** is an
AI-image labelling system for **Lumi**, a fictional photo-sharing app where most posts are real and a few
AI-generated images are posted without a label.

| # | Chapter | What happens | Report link |
| --- | --- | --- | --- |
| 1 | The feed | Social-style feed: guess real or AI, then TrueFrame reveals its calls | Why detection matters |
| 2 | When the label falls off | Post a file or a live webcam photo; metadata label vs pixel detector; "screenshot" strips the metadata | 1.1 |
| 3 | The lazy moderator | "Everything is real" bot vs TrueFrame at real-world rarity | Accuracy paradox, 4.2 |
| 4 | Meet the detectives | SVM, Random Forest, Small CNN, ResNet18: strengths and weaknesses from all 54 models | 4.4 |
| 5 | The rarity dial | Training ratio vs best threshold, with and without each model's fix | 4.3 |
| 6 | Label the uploads | Two-layer labelling of new uploads; threshold/prevalence trade-off curve; Grad-CAM | 4.5, 4.8 |
| 7 | The complaint | TrueFrame's real mistakes, what Lumi should do, and other use cases | 4.6 |
| ⭐ | Bonus: backstage lab | Webcam lab: pick AI posts, record your camera roll, train with/without class weights | 3.5, 4.7 |

Feed posts are held-out test images from the Kaggle subset (never used in training); usernames and captions
are invented. Chapters 4 and 5 read the experiment's results from `data/all_results.csv`.

## Set up once (laptop)

```bash
cd live_app
python -m pip install -r requirements.txt
python prepare_gallery.py "path/to/ipcv_subset.zip" --n 90     # AI images for the bonus lab
python train_best_model.py "path/to/ipcv_subset.zip"           # TrueFrame's detector + feed (a few minutes)
python -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Upload everything in this folder to a public GitHub repository, including `data/`, `ai_gallery/` and
`best_model/` (GitHub's web upload takes 100 files at a time, so upload the folders separately). Then
share.streamlit.io → Create app → repo, branch `main`, file `app.py`, Python 3.12 → Deploy.

## Troubleshooting

- **Chapters say the detector is missing:** run `train_best_model.py`.
- **ResNet18 weights fail to download:** check the internet, or download
  <https://download.pytorch.org/models/resnet18-f37072fd.pth> into this folder.
- **Webcam stream won't start (chapter 6):** close other camera apps, or use *Add photos one at a time*.
