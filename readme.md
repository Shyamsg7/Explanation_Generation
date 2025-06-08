# Explanation Generation in Recommender Systems

This project explores a **two-stage fine-tuning framework** for generating natural language explanations in recommender systems using **causal language models**. The first stage uses supervised fine-tuning (SFT), and the second employs **reinforcement learning (RL)** via **Proximal Policy Optimization (PPO)** to enhance explanation quality.

> 📘 **Full report available:** [`Final_Report.pdf`](./Final_Report.pdf)

---

## 🔍 Problem Statement

Given a prompt describing a user-item interaction (including user ID, item ID, product features, and rating), the goal is to generate a **fluent and helpful explanation** that mimics a real user review.

Example prompt:
```
User Emma Davis (ID: 789) rated the product (ID: 101) 4/5. The product is related to features: nail polish. Write a detailed explanation for this product, highlighting aspects related to nail polish.
```

Expected output:
```
Wow, this is the best deal I’ve seen on nail polish in a long time. You get so many vibrant beautiful colors...
```

---

## 🧠 Methodology

### Stage 1: Supervised Fine-Tuning (SFT)
- Train a causal language model using next-token prediction.
- Dataset: Amazon Reviews (Beauty, Sports, Toys & Games).
- Objective: Learn structure and fluency of user-written reviews.

### Stage 2: Reinforcement Learning (RL)
- Use PPO to fine-tune the model based on semantic similarity between generated and ground-truth explanations.
- Reward signal: Cosine similarity using `SentenceTransformer` embeddings.

---

## 📊 Results

Models like **DeepSeek-R1-Distill-Qwen-1.5B** and **TinyLLaMA** showed significant improvements in BLEU, ROUGE, and **MAUVE** scores after RL-based fine-tuning.

| Model                        | BLEU-4 | ROUGE-1 | ROUGE-2 | MAUVE  |
|-----------------------------|--------|----------|----------|--------|
| GPT-2 (Stage 2)             | ↑      | ↑        | ↑        | ↑      |
| TinyLLaMA (Stage 2)         | ↑↑     | ↑↑       | ↑↑       | ↑↑     |
| DeepSeek-R1 (Stage 2)       | 🔝      | 🔝        | 🔝        | 🔝      |

---

## 🛠️ Project Structure

```
Explanation_Gen/
│
├── data/                  # Preprocessed datasets
├── models/                # Trained checkpoints (Stage 1 and Stage 2)
├── scripts/               # Training and evaluation scripts
│   ├── train_sft.py
│   ├── train_rl.py
│   └── evaluate.py
├── utils/                 # Helper functions (tokenization, reward calculation, etc.)
├── results/               # Metric results, plots, generated outputs
├── config/                # Hyperparameter and model configs
└── README.md              # Project overview (this file)
```

---

## 📦 Setup & Installation

```bash
git clone https://github.com/Shyamsg7/Explanation_Generation.git
cd Explanation_Generation
pip install -r requirements.txt
```

---

## 🚀 Training & Evaluation

### Stage 1: Supervised Fine-Tuning
```bash
python scripts/train_sft.py --config config/gpt2_sft.yaml
```

### Stage 2: RL Fine-Tuning with PPO
```bash
python scripts/train_rl.py --config config/gpt2_rl.yaml
```

### Inference
```bash
python scripts/evaluate.py --model_path models/gpt2_stage2
```

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

## 🧪 Future Directions

- Try larger models (LLaMA-3, Falcon-40B).
- Incorporate human feedback in reward modeling.
- Explore alternative RL algorithms (GRPO, TRPO).

---

## 📄 Citation

If you use this work, please cite:
```
Shyam Sundar Gowda, "Advancing Explanation Generation in Recommender Systems via Supervised and Reinforcement Learning", MTech Project Report, 2025.
```
