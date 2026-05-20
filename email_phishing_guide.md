# Email Phishing Detection Model
## Setup & Training Guide
### Research Group 04 | University of Kelaniya | 2025

---

## 1. What This Pipeline Does

This Python pipeline trains a machine learning model that detects **phishing emails**.
It mirrors the 4-layer architecture of the URL Phishing model, translating URL-based
signals into deep **email header analysis**, **NLP/lexical features**, and
**attachment inspection**.

**Full pipeline:**
1. Load email datasets (CSV files with raw email text or .eml folders)
2. Extract **42 features** per email (structural + header + NLP + attachment)
3. Train XGBoost + LightGBM + Random Forest models
4. Select the best model automatically
5. Export to TFLite format for Android
6. Generate `email_phishing_mobile_meta.json` with scaler parameters

---

## 2. Requirements

### 2.1 Software

| Software | Version | Notes |
|---|---|---|
| Python | 3.9 or higher | Anaconda recommended |
| Jupyter Notebook / JupyterLab | Any recent | Comes with Anaconda |
| Anaconda | Latest | Easiest way to install everything |

### 2.2 Python Libraries

Run this once in the **first cell** of the notebook:

```python
%pip install pandas numpy scikit-learn xgboost lightgbm shap matplotlib seaborn \
             joblib pyarrow tensorflow beautifulsoup4 requests tldextract \
             python-Levenshtein scipy chardet
```

---

## 3. Dataset Setup

You need CSV files containing emails labeled as `phishing` or `legitimate`.
The pipeline supports multiple CSV formats and **auto-detects column names**.

### 3.1 Recommended Datasets

| Dataset Name | Where to Download | Labels |
|---|---|---|
| `phishing_email.csv` | Kaggle — "Phishing Email Detection" | Phishing Email / Safe Email |
| `CEAS_08.csv` | CEAS 2008 spam corpus (GitHub/UCI) | spam / ham |
| `enron_spam_data.csv` | Kaggle — "Enron Email Dataset" | spam / ham |
| `SpamAssassin.csv` | SpamAssassin public corpus | spam / ham |
| `phishtank_emails.csv` | PhishTank exports (manual) | phishing / legitimate |

> **Tip:** Combine multiple datasets using `CSV_PATHS = ["file1.csv", "file2.csv"]`
> in Cell 3 for best accuracy.

### 3.2 How to Configure the CSV Path

Open **Cell 3 (Configuration)** and edit:

```python
# Option A: Single CSV file
CSV_PATH  = "phishing_emails.csv"
CSV_PATHS = None

# Option B: Multiple CSV files
CSV_PATH  = None
CSV_PATHS = ["enron_spam.csv", "ceas08.csv", "phishtank.csv"]

# Option C: Folder of .eml files
EML_DIR   = "emails/"
```

---

## 4. Cell Run Order

Run the cells in this **exact order**. Do NOT skip cells.

| Cell | Name | What it does | Time |
|---|---|---|---|
| Cell 1 | Install Dependencies | Installs all required Python libraries | 2–5 min (first time only) |
| Cell 2 | Imports | Loads all libraries into memory | < 30 sec |
| Cell 3 | Configuration | **SET YOUR CSV PATH HERE** | Instant |
| Cell 4 | Detection Constants | Phishing keywords, trusted domains, Sinhala XAI map | Instant |
| Cell 5 | Feature Extraction | Defines 42 email feature functions | Instant |
| Cell 6 | Feature Cache | Caches features for faster re-runs | Instant |
| Cell 7 | Data Loader | Loads and cleans CSV / .eml data | Instant |
| Cell 8 | Load + Extract Features | Reads emails and builds feature matrix | 5–60 min |
| Cell 9 | Train/Test Split | Splits data 80/20 (stratified) | Instant |
| Cell 10 | Train Models | Trains LightGBM, XGBoost, Random Forest | 2–10 min |
| Cell 10b | Advanced NLP Features | Adds TF-IDF, Levenshtein brand distance | 5–15 min |
| Cell 11 | Evaluate Models | Shows Accuracy, F1, AUC for all models | < 1 min |
| Cell 11b | Threshold Tuning | Finds optimal decision threshold (min 0.70) | < 1 min |
| Cell 11c | Ensemble | Combines top 2 models for better accuracy | < 1 min |
| Cell 11d | Feature Selection | Removes weak features, keeps best 42 | < 1 min |
| Cell 13 | Save Model | Saves `.pkl` model bundle | Instant |
| Cell 13b | Export TFLite | Creates `email_phishing_model.tflite` + JSON | 5–15 min |

---

## 5. The 4-Layer Hybrid Detection Engine

```
Email Input
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 1 — Allowlist                                    │
│  Trusted sender domain + SPF/DKIM/DMARC all pass?       │
│  → YES: Return LEGITIMATE (confidence 0.98) immediately │
└────────────────────────┬────────────────────────────────┘
                         │ NO
                         ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 2 — Hard Rules                                   │
│  Display name spoofing + auth fail? Raw IP in URL?      │
│  Dangerous attachment + auth fail?                      │
│  → YES: Return PHISHING immediately                     │
└────────────────────────┬────────────────────────────────┘
                         │ NO
                         ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 3 — XGBoost ML Classifier                        │
│  P(phishing) >= 0.80 → PHISHING                         │
│  P < 0.80 → Risk calibration based on soft signals      │
│  Risk calibrated score >= 0.60 → PHISHING               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 4 — Sinhala XAI                                  │
│  Generate localized Sinhala explanations for each flag  │
│  SHAP feature importance (if SHAP installed)            │
└─────────────────────────────────────────────────────────┘
```

### Active Defense (URL Extraction)
All URLs embedded in the email body are extracted and passed to the
**URL Phishing Model** (from the URL notebook) for secondary validation.
If any URL is flagged as phishing, the email verdict escalates automatically.

---

## 6. Feature Architecture (42 Features)

| Category | Features |
|---|---|
| **Structural** (15) | `num_urls`, `num_https_urls`, `ratio_https_urls`, `has_raw_ip_url`, `num_tracking_pixels`, `body_text_length`, `body_text_entropy`, `html_to_text_ratio`, `is_html_only`, `subject_length`, `subject_entropy`, `subject_has_re_fwd`, `subject_all_caps_ratio`, `num_received_hops`, `num_http_urls` |
| **Header Anomaly** (10) | `has_spf_fail`, `has_dkim_fail`, `has_dmarc_fail`, `auth_score`, `from_replyto_mismatch`, `is_display_name_spoofed`, `sender_domain_length`, `sender_domain_brand_similarity`, `sender_is_free_email`, `replyto_domain_mismatch` |
| **Attachment** (4) | `num_attachments`, `has_dangerous_attachment`, `num_dangerous_attachments`, `attachment_name_entropy` |
| **NLP/Lexical** (4) | `urgency_keyword_score`, `urgency_keyword_count`, `contains_financial_trigger`, `num_exclamation_marks` |
| **URL-level** (4) | `num_unique_url_domains`, `avg_url_length`, `num_ip_urls`, `num_shortened_urls` |
| **Advanced NLP** (5) | `tfidf_phishing_score`, `domain_entropy`, `levenshtein_brand_min_dist`, `body_word_count`, `num_external_domains` |

---

## 7. Output Files

After running **Cell 13b**, these files are created in the notebook folder:

| File | Size (approx.) | Use |
|---|---|---|
| `email_phishing_model.tflite` | ~22 KB | Copy to Android Studio `assets/` folder |
| `email_phishing_mobile_meta.json` | ~8 KB | Copy to Android Studio `assets/` folder |
| `email_phishing_model_f16.tflite` | ~36 KB | Smaller FP16 version (optional) |
| `email_phishing_model.pkl` | ~5–50 MB | Python use only, not needed for Android |
| `email_feature_cache.parquet` | Varies | Feature cache — speeds up re-runs |

---

## 8. Key Numbers to Know

| Parameter | Value | Meaning |
|---|---|---|
| `n_features` | 42 | Number of email features the model uses |
| `optimal_threshold` | 0.80 | If P(phishing) >= 0.80 → classify as phishing |
| `min_threshold` | 0.70 | Minimum allowed threshold after tuning |
| `model_input_shape` | [1, 42] | Input: 1 email with 42 features |
| `model_output_shape` | [1, 2] | Output: [P(legitimate), P(phishing)] |
| `output_index_phishing` | 1 | `output[0][1]` = phishing probability |
| `output_index_legitimate` | 0 | `output[0][0]` = legitimate probability |
| Expected accuracy | ~90% | On test dataset |

> **Note:** The phishing threshold is 0.80 (stricter than URL model's 0.60)
> because email false positives have higher business impact.

---

## 9. Sinhala XAI Flag Reference

| Flag Key | Sinhala Explanation |
|---|---|
| `spf_fail` | 🔐 SPF සත්‍යාපනය අසාර්ථකයි |
| `dkim_fail` | 🔐 DKIM අත්සන් සත්‍යාපනය අසාර්ථකයි |
| `dmarc_fail` | 🔐 DMARC ප්‍රතිපත්ති සත්‍යාපනය අසාර්ථකයි |
| `display_name_spoof` | 🎭 වංචනික නාමකරණය — ප්‍රසිද්ධ සංස්ථාවක නමක් ව්‍යාජ ලෙස භාවිතා කෙරේ |
| `from_replyto_mismatch` | 📧 'From' හා 'Reply-To' ලිපිනයන් අසමාන වේ |
| `no_https_links` | 🔓 ආරක්ෂිත නොවන සබැඳි (HTTP) අඩංගු වේ |
| `raw_ip_in_url` | 🌐 URL තුළ Raw IP ලිපිනයක් අඩංගු වේ — ඉතා සැකසහිතයි |
| `suspicious_attachment` | ⚠️ අනතුරුදායක ගොනු ඇමිණීමක් (Dangerous Attachment) |
| `high_urgency_score` | 🚨 ඉහළ හදිසි (Urgency) භාෂාව දක්නට ලැබේ |
| `financial_trigger` | 💰 මූල්‍ය ප‍්‍රශ්න ගැන හදිසි ඉල්ලීමක් ඇත |
| `many_urls` | 🔗 බොහෝ සබැඳි (URLs) ඇත |
| `tracking_pixel` | 👁️ සැඟවුණු ගොනු / ලුහු‌බැඳුම් Pixel අඩංගු වේ |
| `ml_model_phishing` | 🤖 ML ආදර්ශකය Phishing ලෙස හඳුනාගෙන ඇත |

---

## 10. Difference Table: URL Model vs Email Model

| Parameter | URL Phishing Model | Email Phishing Model |
|---|---|---|
| Input type | URL string | Raw email / .eml / CSV |
| Features | 42 URL features | 42 email features |
| Feature types | Lexical, structural, DNS | Header, NLP, attachment, auth |
| Phishing threshold | 0.60 | **0.80** (stricter) |
| Min threshold (tuning) | 0.50 | **0.70** |
| Allowlist check | Domain-based | Domain + SPF/DKIM/DMARC |
| Hard rules | IP in URL, entropy | Display name spoof + auth fail |
| XAI language | Sinhala | **Sinhala** (same map style) |
| Android output | `.tflite` + `.json` | `.tflite` + `.json` |

---

## 11. Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` | Library not installed | Run Cell 1 again |
| `FileNotFoundError` | CSV file not found | Check `CSV_PATH` in Cell 3 |
| `ValueError: feature_names mismatch` | Feature count mismatch | Re-run from Cell 10b onwards |
| `chardet` decode error | Unusual email encoding | Already handled — email skipped |
| `FULLY_CONNECTED version error` (Android) | Old TFLite library | Update `tensorflow-lite` to 2.16.1 in `build.gradle` |
| Model not loaded (Android) | Wrong file in assets/ | Re-copy both files from notebook folder |
| Low accuracy (<85%) | Too few training samples | Add more datasets in Cell 3 |

---

## 12. Copying Files to Android

After **Cell 13b** completes successfully:

1. Find the notebook folder on your PC (where the `.ipynb` file is saved)
2. Locate `email_phishing_model.tflite` (~22 KB) and `email_phishing_mobile_meta.json`
3. Open Android Studio → navigate to `app > src > main > assets`
4. Delete the old `email_phishing_model.tflite` and `email_phishing_mobile_meta.json`
5. Copy both new files into the `assets/` folder
6. Click **Run** in Android Studio

> Use the **same `PhishingDetectorHelper.kt`** class from the URL model,
> but change the model filename to `"email_phishing_model.tflite"` and
> the meta filename to `"email_phishing_mobile_meta.json"`.

---

## 13. Quick Demo (No Training Required)

To test the full 4-layer pipeline without training:

```python
# At the bottom of email_phishing_detection.py
run_demo()
```

This runs both a phishing and a legitimate sample through all 4 layers
and prints the HITL JSON output with Sinhala XAI reasons.

---

*Research Group 04 | University of Kelaniya | 2025*
