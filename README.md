## ⚽ **World Cup Scoring Style Clustering using EM + PyTorch**

### 📌 Overview

This project applies **Gaussian Mixture Models (GMMs)** trained using the **Expectation Maximization (EM)** algorithm to discover hidden **goal-scoring styles of World Cup players**. We analyze player performance data from the **2006–2022 FIFA World Cups**, cluster scoring behavior, and interpret each scoring profile.
All machine learning is implemented **from scratch using PyTorch tensors** (no scikit-learn GMM!).

---
Absolutely — here is a professionally formatted **GitHub README** tailored to your World Cup EM project.
Copy–paste into `README.md` in your repo.

---

## ⚽ **World Cup Scoring Style Clustering using EM + PyTorch**

### 📌 Overview

This project applies **Gaussian Mixture Models (GMMs)** trained using the **Expectation Maximization (EM)** algorithm to discover hidden **goal-scoring styles of World Cup players**. We analyze player performance data from the **2006–2022 FIFA World Cups**, cluster scoring behavior, and interpret each scoring profile.
All machine learning is implemented **from scratch using PyTorch tensors** (no scikit-learn GMM!).

---

### 🎯 **Goal**

Identify latent scoring styles such as:

* 🏹 **Shot-heavy strikers**
* 🎯 **Efficient finishers**
* 🧠 **Support attackers**

These representations help with:

* Tactical decision-making
* Player scouting/valuation
* Predictive analytics

---

### 📊 **Dataset**

The dataset includes player stats from multiple FIFA World Cups (2006–2022):

| Feature  | Description                     |
| -------- | ------------------------------- |
| Goals    | Total scored                    |
| Shots    | Attempts                        |
| Minutes  | Time played                     |
| xG       | Expected Goals (chance quality) |
| Position | One-hot encoded (FWD, MID, DEF) |

📌 Players with fewer than **90 minutes played** were excluded.

---

### 🔢 **Algorithm**

We model player styles as a mixture of multivariate Gaussians:

[
p(x) = \sum_{k=1}^{K} \phi_k \cdot \mathcal{N}(x|\mu_k, \Sigma_k)
]

Training is done using **Expectation-Maximization**:

#### 🧮 E-Step

Compute cluster membership probabilities:

[
r_{ik} = \frac{\phi_k \mathcal{N}(x_i|\mu_k,\Sigma_k)}{\sum_{j} \phi_j \mathcal{N}(x_i|\mu_j,\Sigma_j)}
]

#### 📈 M-Step

Update cluster parameters:

[
\mu_k = \frac{1}{N_k} \sum_i r_{ik} x_i,
\qquad
\Sigma_k = \frac{1}{N_k} \sum_i r_{ik}(x_i-\mu_k)(x_i-\mu_k)^T
]

---

### 🛠️ **Technologies**

| Tool       | Usage                    |
| ---------- | ------------------------ |
| Python     | Data + EM implementation |
| PyTorch    | Tensor math, GPU ops     |
| Pandas     | Data handling            |
| Matplotlib | Visualization            |

> 💡 *No scikit-learn clustering was used — EM and GMM are fully implemented using PyTorch.*

---

### 🚀 **How to Run**

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/worldcup-em-clustering.git
cd worldcup-em-clustering
```

Install dependencies (CPU-only PyTorch):

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Run the notebook:

```bash
jupyter notebook
```

---

### 📌 **Project Structure**

```
├── data/                # Dataset (not included if original license restricted)
├── notebooks/
│   └── worldcup_em.ipynb  # Main EM training + analysis
├── src/
│   ├── em_gmm.py          # EM + GMM PyTorch implementation
│   └── utils.py           # Data loading, prep
├── slides/                # Beamer slides (LaTeX)
├── report/                # Final PDF Report
└── README.md              # This file!
```

---

### 📦 **Features**

✔ EM & GMM coded manually using PyTorch
✔ Handles multivariate continuous + categorical features
✔ GPU-accelerated clustering
✔ Real-world World Cup data
✔ Produces interpretable player clusters

---

### 🔍 **Example Output**

Clusters identified:

| Cluster | Style               | Traits                       | Example Players         |
| ------- | ------------------- | ---------------------------- | ----------------------- |
| 1       | Elite Strikers      | High volume shooters         | Mbappé, Ronaldo, Müller |
| 2       | Efficient Finishers | High conversion, few chances | Morata, James Rodríguez |
| 3       | Support Attackers   | Midfield scorers             | Di María, De Bruyne     |

📌 PCA visualization of clusters included in notebook.

---

### 📈 **Possible Extensions**

🚀 Add passing/dribbling stats
🤖 RL simulation of optimal shot choices
🧠 Deep Variational GMM + Player Embeddings
📍 Expand to club data (UEFA, Premier League, La Liga)

---

### 🏆 **Credits**

* FIFA Stats Data
* C. Bishop — *Pattern Recognition and Machine Learning*
* PyTorch Documentation

---

### 📜 **License**

This project is released under the MIT License.

---

If you want, I can also generate:
🔹 `requirements.txt`
🔹 folder templates
🔹 `.gitignore`

Want me to generate them automatically? *(Yes/No)*

### 🎯 **Goal**

Identify latent scoring styles such as:

* 🏹 **Shot-heavy strikers**
* 🎯 **Efficient finishers**
* 🧠 **Support attackers**

These representations help with:

* Tactical decision-making
* Player scouting/valuation
* Predictive analytics

---

### 📊 **Dataset**

The dataset includes player stats from multiple FIFA World Cups (2006–2022):

| Feature  | Description                     |
| -------- | ------------------------------- |
| Goals    | Total scored                    |
| Shots    | Attempts                        |
| Minutes  | Time played                     |
| xG       | Expected Goals (chance quality) |
| Position | One-hot encoded (FWD, MID, DEF) |

📌 Players with fewer than **90 minutes played** were excluded.

---

### 🔢 **Algorithm**

We model player styles as a mixture of multivariate Gaussians:

[
p(x) = \sum_{k=1}^{K} \phi_k \cdot \mathcal{N}(x|\mu_k, \Sigma_k)
]

Training is done using **Expectation-Maximization**:

#### 🧮 E-Step

Compute cluster membership probabilities:

[
r_{ik} = \frac{\phi_k \mathcal{N}(x_i|\mu_k,\Sigma_k)}{\sum_{j} \phi_j \mathcal{N}(x_i|\mu_j,\Sigma_j)}
]

#### 📈 M-Step

Update cluster parameters:

[
\mu_k = \frac{1}{N_k} \sum_i r_{ik} x_i,
\qquad
\Sigma_k = \frac{1}{N_k} \sum_i r_{ik}(x_i-\mu_k)(x_i-\mu_k)^T
]

---

### 🛠️ **Technologies**

| Tool       | Usage                    |
| ---------- | ------------------------ |
| Python     | Data + EM implementation |
| PyTorch    | Tensor math, GPU ops     |
| Pandas     | Data handling            |
| Matplotlib | Visualization            |

> 💡 *No scikit-learn clustering was used — EM and GMM are fully implemented using PyTorch.*

---

### 🚀 **How to Run**

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/worldcup-em-clustering.git
cd worldcup-em-clustering
```

Install dependencies (CPU-only PyTorch):

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Run the notebook:

```bash
jupyter notebook
```

---

### 📌 **Project Structure**

```
├── data/                # Dataset (not included if original license restricted)
├── notebooks/
│   └── worldcup_em.ipynb  # Main EM training + analysis
├── src/
│   ├── em_gmm.py          # EM + GMM PyTorch implementation
│   └── utils.py           # Data loading, prep
├── slides/                # Beamer slides (LaTeX)
├── report/                # Final PDF Report
└── README.md              # This file!
```

---

### 📦 **Features**

✔ EM & GMM coded manually using PyTorch
✔ Handles multivariate continuous + categorical features
✔ GPU-accelerated clustering
✔ Real-world World Cup data
✔ Produces interpretable player clusters

---

### 🔍 **Example Output**

Clusters identified:

| Cluster | Style               | Traits                       | Example Players         |
| ------- | ------------------- | ---------------------------- | ----------------------- |
| 1       | Elite Strikers      | High volume shooters         | Mbappé, Ronaldo, Müller |
| 2       | Efficient Finishers | High conversion, few chances | Morata, James Rodríguez |
| 3       | Support Attackers   | Midfield scorers             | Di María, De Bruyne     |

📌 PCA visualization of clusters included in notebook.

---

### 📈 **Possible Extensions**

🚀 Add passing/dribbling stats
🤖 RL simulation of optimal shot choices
🧠 Deep Variational GMM + Player Embeddings
📍 Expand to club data (UEFA, Premier League, La Liga)

---

### 🏆 **Credits**

* FIFA Stats Data
* C. Bishop — *Pattern Recognition and Machine Learning*