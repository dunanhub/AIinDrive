# tools/eval_folder.py
import json, os, glob, torch
from CarCondition.inference_old import get_model, _tf, DAMAGE_LABELS
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix

def infer_path(p):
    img = Image.open(p).convert("RGB")
    img = ImageOps.exif_transpose(img)
    x = _tf(img).unsqueeze(0)
    with torch.inference_mode():
        m = get_model()
        out = m(x)
        if isinstance(out, (tuple,list)): out = out[0]
        probs = torch.softmax(out, dim=1)[0].cpu()
        return int(torch.argmax(probs).item())

# ожидается структура: data/<label>/*.jpg  (label = имена из DAMAGE_LABELS)
root = "data"
y_true, y_pred = [], []
for i, lbl in enumerate(DAMAGE_LABELS):
    for p in glob.glob(os.path.join(root, lbl, "*")):
        y_true.append(i)
        y_pred.append(infer_path(p))

print(classification_report(y_true, y_pred, target_names=DAMAGE_LABELS, digits=4))
print(confusion_matrix(y_true, y_pred))
