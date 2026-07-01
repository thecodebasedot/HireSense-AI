# HireSense AI 🧠💼

**চাকরির আবেদনকারী স্ক্রিনিং সিস্টেম — Support Vector Machine (SVM) দিয়ে তৈরি**

HireSense AI একটি মেশিন লার্নিং ভিত্তিক সিস্টেম যা চাকরির আবেদনকারীদের
স্বয়ংক্রিয়ভাবে স্ক্রিনিং করে — প্রার্থীর অভিজ্ঞতা, শিক্ষা, দক্ষতা, ইন্টারভিউ
স্কোর ইত্যাদি বিশ্লেষণ করে সিদ্ধান্ত দেয় প্রার্থীকে **শর্টলিস্ট** করা হবে নাকি
**বাতিল**।

> An SVM-based system that automatically screens job applicants and predicts
> whether each candidate should be **shortlisted** or **rejected**.

---

## 🎯 প্রজেক্ট সম্পর্কে (Overview)

| বিষয় | বিবরণ |
|------|-------|
| **Project Name** | HireSense AI |
| **Problem Solved** | চাকরির আবেদনকারীদের স্ক্রিনিং (Job Applicant Screening) |
| **ML Algorithm** | Support Vector Machine (SVM) |
| **Type** | Binary Classification (Shortlist / Reject) |

কেন SVM? আবেদনকারীর ফিচারগুলো সংখ্যাগত এবং শ্রেণিগুলোর মধ্যে সীমানা অ-রৈখিক
হতে পারে। SVM কার্নেল ট্রিকের মাধ্যমে এমন সীমানা ভালোভাবে শিখতে পারে এবং
মাঝারি আকারের ডেটাসেটে দক্ষ।

---

## 📊 ফিচার (Features)

মডেলটি প্রতিটি আবেদনকারীর নিচের ৮টি বৈশিষ্ট্য ব্যবহার করে:

| ফিচার | অর্থ | রেঞ্জ |
|-------|------|-------|
| `years_experience` | প্রাসঙ্গিক কাজের অভিজ্ঞতা (বছর) | 0–25 |
| `education_level` | শিক্ষাগত স্তর (0=HS, 1=Bachelor, 2=Master, 3=PhD) | 0–3 |
| `skill_match_score` | চাকরির প্রয়োজনের সাথে দক্ষতার মিল | 0–100 |
| `interview_score` | ইন্টারভিউ পারফরম্যান্স | 0–100 |
| `communication_score` | যোগাযোগ দক্ষতা | 0–100 |
| `num_certifications` | প্রাসঙ্গিক সার্টিফিকেশন সংখ্যা | 0–10 |
| `num_projects` | সম্পন্ন প্রজেক্ট সংখ্যা | 0–20 |
| `gpa` | সিজিপিএ | 2.0–4.0 |

**Target:** `shortlisted` → `1` (শর্টলিস্ট) অথবা `0` (বাতিল)

---

## 🗂️ প্রজেক্ট স্ট্রাকচার (Structure)

```
HireSense-AI/
├── src/
│   ├── config.py         # ফিচার স্কিমা, পাথ, ধ্রুবক
│   ├── generate_data.py  # সিন্থেটিক আবেদনকারী ডেটাসেট জেনারেটর
│   ├── train.py          # SVM ট্রেনিং + হাইপারপ্যারামিটার টিউনিং
│   └── predict.py        # নতুন আবেদনকারী স্ক্রিনিং (single/batch)
├── data/
│   └── sample_applicants.csv   # উদাহরণ ইনপুট (batch scoring-এর জন্য)
├── models/               # ট্রেইনড মডেল এখানে সেভ হয়
├── tests/
│   └── test_pipeline.py  # বেসিক টেস্ট
├── run.sh                # এক কমান্ডে পুরো পাইপলাইন
└── requirements.txt
```

---

## 🚀 কীভাবে চালাবেন (Quick Start)

### ১. সবকিছু এক কমান্ডে

```bash
bash run.sh
```

এটি ডিপেন্ডেন্সি ইনস্টল করবে → ডেটা তৈরি করবে → মডেল ট্রেন করবে → ডেমো
প্রেডিকশন দেখাবে।

### ২. ধাপে ধাপে (Step by step)

```bash
# ডিপেন্ডেন্সি ইনস্টল
pip install -r requirements.txt

# ১) সিন্থেটিক ডেটাসেট তৈরি
python src/generate_data.py            # ডিফল্ট ২০০০ আবেদনকারী
python src/generate_data.py -n 5000    # কাস্টম সংখ্যা

# ২) SVM মডেল ট্রেন
python src/train.py

# ৩) একজন আবেদনকারী স্ক্রিন করুন
python src/predict.py \
    --years-experience 6 --education-level 2 \
    --skill-match-score 85 --interview-score 80 \
    --communication-score 78 --num-certifications 3 \
    --num-projects 8 --gpa 3.7

# ৪) CSV থেকে একাধিক আবেদনকারী স্ক্রিন করুন
python src/predict.py --csv data/sample_applicants.csv
```

---

## 🧪 টেস্ট চালানো (Run tests)

```bash
python tests/test_pipeline.py
# অথবা pytest থাকলে:
python -m pytest tests/ -q
```

---

## ⚙️ মডেল কীভাবে কাজ করে (How it works)

1. **Preprocessing** — `StandardScaler` দিয়ে সব ফিচার স্ট্যান্ডার্ডাইজ করা হয়
   (SVM দূরত্ব-নির্ভর, তাই স্কেলিং জরুরি)।
2. **Model** — `SVC` (RBF ও linear কার্নেল), `GridSearchCV` দিয়ে `C`, `gamma`,
   কার্নেল টিউন করা হয় ৫-fold ক্রস-ভ্যালিডেশনে (`roc_auc` স্কোরিং)।
3. **Calibration** — `CalibratedClassifierCV` দিয়ে নির্ভরযোগ্য সম্ভাবনা
   (probability) পাওয়া যায়, যাতে প্রতিটি সিদ্ধান্তের সাথে একটি কনফিডেন্স স্কোর
   দেখানো যায়।
4. **Output** — প্রতিটি আবেদনকারীর জন্য সিদ্ধান্ত (`Shortlist`/`Reject`) এবং
   শর্টলিস্ট হওয়ার সম্ভাবনা।

### 📈 উদাহরণ পারফরম্যান্স (Sample metrics)

সিন্থেটিক ২০০০-স্যাম্পল ডেটাসেটে (হেল্ড-আউট টেস্ট সেট):

```
Accuracy : ~0.82
ROC-AUC  : ~0.90
```

> নোট: এখানে আসল হায়ারিং ডেটা ব্যবহার করা হয়নি (সংবেদনশীল ও গোপনীয়)। এর
> পরিবর্তে বাস্তবসম্মত সিন্থেটিক ডেটা তৈরি করা হয়েছে যাতে পুরো পাইপলাইন
> রিপ্রোডিউসিবলভাবে চালানো যায়। আসল ডেটা `data/applicants.csv` ফরম্যাটে
> (একই কলাম) দিলে সরাসরি `python src/train.py` চালানো যাবে।

---

## 📄 লাইসেন্স (License)

এই প্রজেক্টটি রিপোজিটরির [LICENSE](LICENSE) ফাইলের অধীনে প্রকাশিত।
