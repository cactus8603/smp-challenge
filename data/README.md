# External Data Collection for SMP Challenge

## Overview

This directory contains externally collected data from Flickr, designed to **augment the SMP Challenge dataset**.

The goal is to:

* Increase data diversity
* Improve model generalization
* Provide additional supervision via pseudo labels

All data is crawled using the Flickr public API with **license filtering** and converted into **SMP-compatible format**.

---

## Key Design Principles

### 1. Distribution Matching

Instead of randomly collecting data, we **align the external data distribution with the SMP training set**.

We first analyze the category distribution in the SMP training data:

| Category             | Ratio (%) |
| -------------------- | --------- |
| Travel&Active&Sports | 21.75     |
| Holiday&Celebrations | 17.92     |
| Fashion              | 16.13     |
| Entertainment        | 9.68      |
| Social&People        | 7.64      |
| Whether&Season       | 6.64      |
| Animal               | 6.54      |
| Food                 | 5.47      |
| Urban                | 5.00      |
| Electronics          | 1.84      |
| Family               | 1.40      |

---

### 2. Text-based Crawling Strategy

Since Flickr API does not provide SMP category labels, we use:

> **Text queries (`--text`) to approximate category distribution**

Each category is mapped to a representative query:

| Category             | Query         |
| -------------------- | ------------- |
| Travel&Active&Sports | `travel`      |
| Holiday&Celebrations | `holiday`     |
| Fashion              | `fashion`     |
| Entertainment        | `concert`     |
| Social&People        | `people`      |
| Whether&Season       | `winter`      |
| Animal               | `animal`      |
| Food                 | `food`        |
| Urban                | `city`        |
| Electronics          | `electronics` |
| Family               | `family`      |

We then allocate crawling budget according to the SMP distribution.

---

### 3. Data Organization

Each category is stored in a separate directory:

```
category_crawls/
├── extra_data_Travel_Active_Sports/
├── extra_data_Holiday_Celebrations/
├── extra_data_Fashion/
...
```

Each folder contains:

* `extra_text.jsonl`
* `extra_temporalspatial_information.jsonl`
* `extra_user_data.jsonl`
* `extra_additional_information.jsonl`
* `extra_category.jsonl`
* `extra_pseudo_label.jsonl`
* `extra_img_filepath.txt`
* `images/`

---

### 4. Pseudo Label Definition

Since the official SMP popularity label is unavailable, we define:

```
PseudoLogViewScore = log(1 + views)
```

Additional metadata:

* `views`: raw view count
* `post_age_days`: time since posting

⚠️ Note:

* This is **not the official SMP label**
* It is only used for **pretraining / auxiliary learning**

---

### 5. Image Collection

Images are downloaded during crawling to avoid:

* broken URLs
* missing data
* alignment issues

Each image path corresponds to:

```
extra_img_filepath.txt
```

---

## Usage

### Run crawler (PowerShell)

```
powershell -ExecutionPolicy Bypass -File .\run_smp_category_crawl.ps1
```

### Adjust total dataset size

```
$env:TOTAL_ITEMS=50000
powershell -ExecutionPolicy Bypass -File .\run_smp_category_crawl.ps1
```

---

## Notes

* External data is used for:

  * representation learning
  * feature augmentation
  * ranking pretraining

* It is **not directly merged with SMP labels** to avoid distribution mismatch.

---

## Summary

This dataset is built with a **data-centric approach**:

* Match distribution instead of random crawling
* Use text queries to approximate semantic categories
* Maintain SMP-compatible structure
* Provide scalable external supervision

---

## TODO

* [ ] Image quality filtering
* [ ] Duplicate removal (hash-based)
* [ ] Category refinement via model prediction
* [ ] Better query expansion per category

---
