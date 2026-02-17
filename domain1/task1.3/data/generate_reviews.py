# Generate customer review data 

import json
import random
import datetime
import uuid 
import pathlib
import pandas as pd


N = 20
products = ["P-1001","P-1002","P-1003","P-1004"]
customers = ["C-2001","C-2002","C-2003","C-2004","C-2005"]


def rand_date(start="2025-01-01", end="2026-01-31"):
    s = datetime.date.fromisoformat(start)
    e = datetime.date.fromisoformat(end)
    delta = (e - s).days
    d = s + datetime.timedelta(days=random.randint(0, delta))
    return d.isoformat()  # YYYY-MM-DD


def make_review():
    pid = random.choice(products)
    cid = random.choice(customers)
    rating = str(random.randint(1,5))  # keep as string to satisfy regex
    # >= 10 chars, <= 5000
    texts = [
        "Absolutely love this product - met my expectations.",
        "Good value for money and quick delivery.",
        "Quality was acceptable; packaging could be better.",
        "Exceeded expectations; will buy again soon!",
        "Decent product; customer support was responsive."
        "The produt was aweful. I was not worth the buck."
    ]
    review_text = random.choice(texts)
    return {
        "review_id": str(uuid.uuid4()),
        "product_id": pid,
        "customer_id": cid,
        "rating": rating,
        "review_date": rand_date(),
        "review_text": review_text
    }


output_file_name = "product_reviews.jsonl"

pathlib.Path(output_file_name).write_text(
    "\n".join(json.dumps(make_review()) for _ in range(N)),
    encoding="utf-8"
)
print(f"Wrote {output_file_name}")


# rows = [json.loads(l) for l in open(output_file_name,"r",encoding="utf-8")]
# df = pd.DataFrame(rows)
# df.to_parquet("product_reviews.parquet", index=False)
# print("Wrote product_reviews.parquet")