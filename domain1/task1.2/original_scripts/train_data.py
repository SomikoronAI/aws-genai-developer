# Assignment Part 4

import pandas as pd
import json

# Create a financial Q&A dataset
data = [
    {"question": "What is a 401(k)?", "answer": "A 401(k) is a tax-advantaged retirement savings plan offered by employers."},
    {"question": "How does compound interest work?", "answer": "Compound interest is when you earn interest on both the money you've saved and the interest you earn."},
    # Add more examples...
]

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV for SageMaker training
df.to_csv("financial_qa_dataset.csv", index=False)
