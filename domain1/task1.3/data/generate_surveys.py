# Generate customer review survey data 

import csv
import random
import datetime


# Configuration parameters 
NUM_ROWS = 100
OUTPUT_FILE = "surveys.csv"

CUSTOMERS = ["C-200"+str(x) for x in range(25)] #["C-2001","C-2002","C-2003","C-2004","C-2005"]
PRODUCTS  = ["P-100"+str(x) for x in range(5)]  #["P-1001","P-1002","P-1003","P-1004", ]

COMMENTS  = [
    "Great product and good value.",
    "Average experience overall.",
    "Could be better; packaging was weak.",
    "Very satisfied with the purchase.",
    "Neutral experience; works fine.",
    "Product in the web and the product at hand does not match.",
    "There is always a hidden fee.",
    "So far, I am okay with the service. Could do more to get better."
    "Quality is decent, but shipping took too long.",
    "Not bad, but I expected more for the price.",
    "Excellent customer support; solved my issue quickly.",
    "The product works, but the instructions were unclear.",
    "Exceeded my expectations in every way.",
    "Feels cheaper than advertised, bought only yesterday and already malfunctioning.",
    "Service was smooth, though delivery updates were confusing.",
    "Pretty good overall; minor flaws but nothing serious.",
    "The item arrived sooner than expected - a pleasant surprise.",
    "Performs as advertised, no major complaints.",
    "Poor quality and the color is different than shown.",
    "Okay with the purchase; would think twice before buying again."
]

IIMPROVEMENTS = [
    "The online chatbot can be improved further to provide more accurate responses.",
    "Customer service needs more human agents during peak hours.",
    "Packaging should be strengthened to better protect fragile electronic items.",
    "Delivery time estimates should be more accurate and reliable.",
    "The mobile app could be optimized to load faster on older devices.",
    "Product descriptions on the website should include more detailed specifications.",
    "Return and refund processes should be made simpler and faster.",
    "Customer support should follow up after resolving reported issues.",
    "The checkout process could be streamlined to reduce abandoned carts.",
    "More payment options should be added for international customers.",
    "Product quality checks should be improved before shipping.",
    "Order tracking updates should be more frequent and transparent.",
    "Customer service representatives need better product training.",
    "The website search functionality should return more relevant results.",
    "Instructions included with the product should be clearer and more detailed.",
    "Live chat support availability should be extended beyond business hours.",
    "The loyalty rewards program could offer more meaningful benefits.",
    "Email notifications should be better timed and less repetitive.",
    "The product packaging design could be more environmentally friendly.",
    "Customer feedback should be acknowledged more promptly.",
    "Warranty information should be easier to find and understand.",
    "The onboarding experience for new users could be improved.",
    "More proactive communication is needed when delays occur.",
    "Product images should better reflect the actual item received.",
    "After-sales support could be more responsive and consistent."
]


def random_date(start="2025-01-01", end="2026-01-31"):
    s = datetime.date.fromisoformat(start)
    e = datetime.date.fromisoformat(end)
    delta = (e - s).days
    return (s + datetime.timedelta(days=random.randint(0, delta))).isoformat()


with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["survey_id", "customer_id", "product_id", "survey_date",
                "overall_satisfaction", "product_rating", "service_rating", 
                "comments", "improvement_area"])

    for i in range(NUM_ROWS):
        cid = random.choice(CUSTOMERS)
        pid = random.choice(PRODUCTS)
        row = [
            f"S-{10000+i}",
            cid,
            pid,
            random_date(),
            random.randint(1,5),
            random.randint(1,5),
            random.randint(1,5),
            random.choice(COMMENTS),
            random.choice(IIMPROVEMENTS)
        ]
        w.writerow(row)

print(f"Created {OUTPUT_FILE} with {NUM_ROWS} rows.")
