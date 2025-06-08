# Explanation Generation in Recommender Systems

This project explores a **two-stage fine-tuning framework** for generating natural language explanations in recommender systems using **causal language models**. The first stage uses supervised fine-tuning (SFT), and the second employs **reinforcement learning (RL)** via **Proximal Policy Optimization (PPO)** to enhance explanation quality.



---

## 🔍 Problem Statement

Given a prompt describing a user-item interaction (including user ID, item ID, product features, and rating), the goal is to generate a **fluent and helpful explanation** that mimics a real user review.


## 🧠 Methodology

### Stage 1: Supervised Fine-Tuning (SFT)
- Train a causal language model using next-token prediction.
- Dataset: Amazon Reviews (Beauty, Sports, Toys & Games).
- Objective: Learn structure and fluency of user-written reviews.

### Stage 2: Reinforcement Learning (RL)
- Use PPO to fine-tune the model based on semantic similarity between generated and ground-truth explanations.
- Reward signal: Cosine similarity using `SentenceTransformer` embeddings.

---



## 📈 Evaluation Metrics

- **BLEU-4**
- **ROUGE-1/2**
- **MAUVE** (semantic and distributional similarity)

> All reported scores are scaled to [0–100] for interpretability.

---

## 📌 Key Takeaways

- Two-stage training significantly improves explanation quality.
- RL-based fine-tuning enhances semantic alignment and fluency.
- MAUVE is a valuable metric for measuring human-likeness in generated text.

---

