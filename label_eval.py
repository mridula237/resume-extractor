import json
import os
from openai import OpenAI
from schema import Resume

gpt = OpenAI()
os.makedirs("eval/gold", exist_ok=True)

for fname in sorted(os.listdir("eval/resumes")):
    stem = fname.replace(".txt", "").replace(".pdf", "")
    out = f"eval/gold/{stem}.json"

    if os.path.exists(out):
        print(f"skip {fname}")
        continue

    print(f"labeling {fname}...")
    with open(f"eval/resumes/{fname}") as f:
        text = f.read()

    r = gpt.beta.chat.completions.parse(
        model="gpt-5.4-nano",
        max_completion_tokens=1500,
        messages=[
            {"role": "system", "content": "Extract structured resume data accurately and completely."},
            {"role": "user", "content": text}
        ],
        response_format=Resume,
    )
    gold = json.loads(r.choices[0].message.content)
    with open(out, "w") as f:
        json.dump(gold, f, indent=2)
    print(f"  saved {out}")

print("Done.")