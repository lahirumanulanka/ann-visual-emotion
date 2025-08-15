# 📊 Exploratory Data Analysis (EDA) — Emotion Recognition Dataset

## 1. Dataset Overview
The dataset is structured into three splits:

- **Training set:** 13,864 images  
- **Validation set:** 1,734 images  
- **Test set:** 1,734 images  

Each image is labeled with one of **8 emotion classes** as defined in `label_map.json`:

| Emotion       | Label ID |
|---------------|----------|
| Amusement     | 0        |
| Anger         | 1        |
| Awe           | 2        |
| Contentment   | 3        |
| Disgust       | 4        |
| Excitement    | 5        |
| Fear          | 6        |
| Sadness       | 7        |

---

## 2. Class Distribution

### Training Set
| Emotion     | Count |
|-------------|-------|
| Amusement   | 3,046 |
| Anger       | 1,119 |
| Awe         | 491   |
| Contentment | 2,512 |
| Disgust     | 21    |
| Excitement  | 4,910 |
| Fear        | 362   |
| Sadness     | 1,403 |

### Validation Set
| Emotion     | Count |
|-------------|-------|
| Amusement   | 381 |
| Anger       | 140 |
| Awe         | 62  |
| Contentment | 314 |
| Disgust     | 3   |
| Excitement  | 614 |
| Fear        | 45  |
| Sadness     | 175 |

### Test Set
| Emotion     | Count |
|-------------|-------|
| Amusement   | 381 |
| Anger       | 140 |
| Awe         | 61  |
| Contentment | 314 |
| Disgust     | 3   |
| Excitement  | 614 |
| Fear        | 45  |
| Sadness     | 176 |

---

## 3. Observations
- **Class Imbalance:**  
  - "Excitement" and "Amusement" dominate the dataset.  
  - "Disgust" is severely underrepresented (only 27 total images across all splits).  
- **Potential impact:** Models may be biased towards predicting majority classes.  
  - Solution: Consider **class weights** in loss function or **oversampling** for minority classes.

---

## 4. Dataset Quality Checks
- **File existence:** Verified that all image paths listed in the CSV files are valid.  
- **Image mode:** All images converted to RGB for consistency.  
- **Image size:** Will be resized to `(224, 224)` before training.

---

## 5. Data Preprocessing Steps
1. **Resize** all images to a fixed size (224×224 pixels).
2. **Data augmentation** for the training set:
   - Random horizontal flips
   - Random rotations (±10 degrees)
   - Color jitter (brightness, contrast, saturation)
3. **Normalization** using ImageNet mean & std for compatibility with CNN training:
   - Mean: `[0.485, 0.456, 0.406]`
   - Std: `[0.229, 0.224, 0.225]`
