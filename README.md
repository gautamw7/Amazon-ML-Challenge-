# Amazon-ML-Challenge- – Price Prediction (v1.0.0)

**TL;DR**: End‑to‑end baseline for product **price prediction** using product text (`catalog_content`) + engineered features.  
Pipeline includes **leakage removal**, **unit/value parsing**, **pack detection**, TF‑IDF + SVD embeddings, and an **XGBoost** regressor optimized on **SMAPE**-friendly setup (log target + multiple diagnostics).

## At the end of the Hackathon I got 56.87 SMAPE on the test data, with the following things been done: Feature Engineering, Data Cleaning, Data Preprocessing, TF-IDF, XGBoost and using only the textual, categorical and numerical data. Preivously before XGBoost GBR was used, with acheiving 59.69 SMAPE on the test. 
## The Repo serves the purpose to logging for further iteration of the solution as pratice and curiosty. Thanks for checking it !! Considering I haven't used visual data.

---

## 🧩 Problem

Predict product `price` given:
- `sample_id`
- `catalog_content` (title + description + IPQ)
- `image_link` (not used in v1 baseline)
- `price` (only in train)

**Evaluation**: SMAPE (Symmetric Mean Absolute Percentage Error).  
Lower is better; bounded in \[0, 200\].

---

## 📦 Dataset

- **Train**: 75k rows with labels (`price`)
- **Test** : 75k rows without labels  
- Output: CSV with `sample_id` and `price` (positive float)

> ⚠️ Dataset is **not** included. If it’s proprietary, do not upload it to the repo. Place files in `data/` locally.

---

## 🏗️ Approach Overview

**1) Data Sanity & Leakage**
- Duplicate checks (`image_link`, `catalog_content`)
- Basic URL validation
- **Price leakage removal** inside `catalog_content`
- Column integrity checks

**2) Field Extraction & Normalization**
- Parse **Item Name**, **Value**, **Unit** from `catalog_content`
- Normalize units (e.g., `fluid ounce` → `fl oz`, `grams` → `g`, etc.)
- Fallback regex to detect patterns like `6.7 oz`, `2 kg` when explicit fields are missing
- Cleaned item name (`item_name_clean`) removing pack/size suffixes

**3) Pack Handling**
- Detect `is_pack_item` via patterns (`pack of N`, `N count`, `N ct`, `N pack`, etc.)
- Derive `pack_count` and compute `price_per_pack`

**4) Text Features**
- TF‑IDF (1‑2 grams) on:
  - `item_name_clean`
  - `cleaned_catalog_content` (value/unit/price tokens removed)
- Dimensionality reduction with **TruncatedSVD** (256 dims each)

**5) Tabular + Text Fusion**
- Numeric: `value`, `pack_count`, `is_pack_item`
- Categorical: `unit` (one‑hot)
- Text: TF‑IDF → SVD embeddings (name + desc)
- **Model**: XGBoost (tree_method=`hist`), log‑target training (`y = log1p(price)`)

**6) Optional LLM augmentation (Gemini)**
- For a small subset with missing `item_name_clean` or `value`, prompt Gemini to infer missing fields from `catalog_content`.  
- This is **optional**, behind an API key, and can be disabled.

---

## 📊 Metrics (local hold‑out)

The pipeline reports:
- **RMSE**, **MAE**, **R²**
- **SMAPE**, **MAPE**  
(Prices are evaluated **after** inverting the log transform.)




