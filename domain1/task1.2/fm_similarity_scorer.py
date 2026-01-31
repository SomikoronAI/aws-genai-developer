# Assignment Part 1
"""
QA Evaluation Metrics

This script provides functions to evaluate the quality of predicted answers
against ground-truth answers in Question Answering (QA) tasks. It computes
the following metrics:

- Exact Match (EM): 1 if normalized prediction matches the ground truth exactly, else 0.
- F1 Score: Token-level overlap between prediction and ground truth.
- ROUGE-L: Longest common subsequence-based F1 score for longer answers.
- BLEU: n-gram precision-based score to measure similarity of word sequences.

Normalization includes lowercasing, removing punctuation and articles, and
trimming whitespace to ensure fair comparison.

Dependencies:
- nltk
- rouge-score
pip install nltk rouge-score

Example usage:
ground_truth = "A 401(k) is a tax-advantaged retirement savings plan offered by employers."
prediction = "A 401(k) is a retirement savings plan."
metrics = evaluate_qa_metrics(prediction, ground_truth)
print(metrics)
"""


import re
import string
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


# -----------------------------
# Normalization helper
# -----------------------------
def normalize_answer(s):
    """Lower text and remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def white_space_fix(text):
        return ' '.join(text.split())

    s = s.lower()
    s = remove_articles(s)
    s = remove_punctuation(s)
    s = white_space_fix(s)
    return s

# -----------------------------
# Exact Match
# -----------------------------
def exact_match_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

# -----------------------------
# F1 Score
# -----------------------------
def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

# -----------------------------
# ROUGE-L Score
# -----------------------------
def rouge_l_score(prediction, ground_truth):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(ground_truth, prediction)
    return scores['rougeL'].fmeasure  # F1-score of ROUGE-L

# -----------------------------
# BLEU Score
# -----------------------------
def bleu_score(prediction, ground_truth):
    # tokenize by whitespace
    reference = normalize_answer(ground_truth).split()
    candidate = normalize_answer(prediction).split()
    # smoothing prevents zero for short sentences
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference], candidate, smoothing_function=smoothie)

# -----------------------------
# Unified function
# -----------------------------
def evaluate_qa_metrics(prediction, ground_truth):
    """Returns a dictionary with EM, F1, ROUGE-L, BLEU"""
    return {
        'EM': exact_match_score(prediction, ground_truth),
        'F1': f1_score(prediction, ground_truth),
        'ROUGE-L': rouge_l_score(prediction, ground_truth),
        'BLEU': bleu_score(prediction, ground_truth)
    }

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    ground_truth = "A 401(k) is a tax-advantaged retirement savings plan offered by employers."
    prediction = "A 401(k) is a retirement savings plan."
    
    metrics = evaluate_qa_metrics(prediction, ground_truth)
    print(metrics)
