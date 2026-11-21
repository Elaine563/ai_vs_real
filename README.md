# Catch the Rare Fake: live class-imbalance demo

A Teachable-Machine-style Streamlit app for the XBDS3024N assignment. You choose how rare the
AI-generated images are, record the real photos with your webcam, and train the same four model
families as the report (SVM, Random Forest, Small CNN, ResNet18), each with **no handling** and with
**class weights**, so you can watch the accuracy paradox happen live.

## Run it on your laptop

```bash
cd live_app
python -m pip install -r requirements.txt
python prepare_gallery.py path/to/ipcv_subset.zip     # fills ai_gallery/ with 150 AI images (once)
streamlit run app.py
```

The browser opens at `http://localhost:8501`. Allow camera access when asked.
The first training run downloads the ResNet18 ImageNet weights (about 45 MB, once).

## How to use it

Use the sidebar menu (the `>` button on a phone or narrow window) to move between steps.

1. **Pick AI images.** Tick a few gallery images or press *Pick 5 at random*. These are the minority
   class. Images you don't pick become the AI half of the test set.
2. **Record real photos.** Set the ratio in the sidebar. The app needs `ratio × AI images` training
   photos plus 20 test photos. Press **START** on the video, then **● Record**. It captures 5 frames a
   second and stops by itself at the target. Move around while it records.
3. **Train & compare.** Trains every model twice (about 10 to 60 seconds on a laptop CPU). The chart
   compares accuracy with AI recall; a red banner appears if a model caught no AI images.
4. **Test it live.** Take a webcam snapshot, upload an image, or step through the test images and see
   how each model votes.

## How it maps to the report

| Report | This app |
| --- | --- |
| 128 × 128 centre crop, 1,846 hand-crafted features | Same code |
| SVM (RBF, C = 10), Random Forest (300 trees) | Same settings |
| Small CNN from scratch | Same architecture, 64 px input, 10 epochs (CPU) |
| ResNet18 fine-tuned on a T4 GPU | ImageNet ResNet18 kept frozen, new output layer trained (CPU) |
| Class weights / weighted loss `n / (2 × count)` | Same formula |
| Fixed 10:1 test set | 20 new webcam frames + up to 20 unpicked AI images; accuracy also shown re-weighted to your ratio |

**Be honest about the limits if you demo this.** Webcam frames are very similar to each other, and your room
looks very different from AI art, so the task is easier than in the report. Recording paintings, posters,
phone screens or printed photos makes it harder and more realistic.

## Troubleshooting

- **Video stays black / START does nothing:** close other apps using the camera (Zoom, Teams). Use
  Chrome. Or use the *Add photos one at a time* section below the video.
- **Gallery is empty:** run `prepare_gallery.py`, or upload AI images on page 1.
- **Streamlit Community Cloud:** the webcam stream needs a TURN server there, and the free tier is tight
  on memory for PyTorch. Run it locally for the demo.
