"""Generate a track with Google Lyria through the Gemini API.

    GEMINI_API_KEY=... python tools/gemini_music.py --prompt "..." --out music/track [--model lyria-3.5]

Writes <out>_0.mp3 (and any text the model returns, e.g. lyrics/structure, to <out>.txt).
Then: python -m synthviz --audio music/track_0.mp3 --window 31.5-90.6 --style future_bass --out out/v.mp4
"""
import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--prompt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="lyria-3.5")
a = ap.parse_args()

body = {"contents": [{"parts": [{"text": a.prompt}]}]}
req = urllib.request.Request(
    f"https://generativelanguage.googleapis.com/v1beta/models/{a.model}:generateContent",
    data=json.dumps(body).encode(),
    headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"], "Content-Type": "application/json"})
try:
    r = json.load(urllib.request.urlopen(req, timeout=600))
except urllib.error.HTTPError as e:
    print(e.read().decode()[:2000]); sys.exit(1)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
n, texts = 0, []
for c in r.get("candidates", []):
    for p in c["content"]["parts"]:
        if "inlineData" in p:
            ext = p["inlineData"]["mimeType"].split("/")[-1].replace("mpeg", "mp3")
            path = f"{a.out}_{n}.{ext}"
            open(path, "wb").write(base64.b64decode(p["inlineData"]["data"]))
            print("saved", path); n += 1
        elif "text" in p:
            texts.append(p["text"])
if texts:
    open(a.out + ".txt", "w").write("\n".join(texts))
print(json.dumps(r.get("usageMetadata", {})))
