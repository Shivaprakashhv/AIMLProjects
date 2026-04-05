# AI ML Projects

A comprehensive collection of **AI & Machine Learning projects** covering end-to-end pipelines from data analysis to advanced deep learning techniques.

## 🎯 Project Overview

This repository demonstrates practical implementations of:
- **E-commerce & Recommendation Systems** (Amazon products, Click-through rate prediction)
- **Sports Analytics** (IPL data analysis & insights)
- **Advanced NLP & LLM Techniques** (LoRA fine-tuning, RAG pipelines)
- **Computer Vision** (License plate detection with YOLOv8)
- **Sales & Marketing Analytics** (Campaign performance prediction, customer retention)

**Tech Stack**: Python 3.8+, PyTorch, Scikit-learn, LangChain, YOLOv8, OpenCV, Milvus

---

## 📁 Project Files & Solutions

### 1. **LoRA Fine Tuning.py**
- **Steps**: Load model → Configure LoRA → Prepare dataset → Train → Save adapter
- **Problem**: Efficiently fine-tune large LLMs with minimal memory using parameter-efficient LoRA
- **Use Case**: Adapt large language models to domain-specific tasks without full retraining

### 2. **RAG_Pipeline.py**
- **Steps**: Load data → Chunk documents → Generate embeddings → Store in Milvus → Retrieve & Generate answers
- **Problem**: Build semantic search + LLM system to answer questions from documents
- **Use Case**: Document-based Q&A systems, knowledge base retrieval, enterprise search

### 3. **Click_Through_Rate_Prediction.ipynb**
- **Steps**: EDA → Handle imbalance → Extract temporal features → Train model → Predict clicks
- **Problem**: Predict ad clicks in highly imbalanced dataset (93% negative class)
- **Domain**: E-commerce & Digital Marketing
- **Techniques**: Class balancing, feature engineering, gradient boosting

### 4. **Amazon_Product_Recommendation.ipynb**
- **Steps**: Load data → EDA → Extract features → Build recommendation model
- **Problem**: Build product recommendation system from Amazon reviews & ratings data
- **Domain**: E-commerce & Recommendation Systems
- **Outcome**: Personalized product suggestions improving customer engagement

### 5. **IPL_Data_Analysis.ipynb**
- **Steps**: Download data → Clean team names → EDA → Extract patterns → Visualize trends
- **Problem**: Analyze cricket performance metrics across IPL seasons (2008-2024)
- **Domain**: Sports Analytics
- **Insights**: Team performance, player statistics, seasonal trends, match outcomes

### 6. **E_commerce_Marketing_and_Sales.ipynb**
- **Steps**: Analyze acquisitions → Monthly trends → Retention cohort analysis
- **Problem**: Identify seasonal patterns and improve customer retention strategies
- **Domain**: E-commerce Analytics & Marketing
- **Metrics**: Customer acquisition cost, lifetime value, churn analysis

### 7. **Predicting_Sales_from_Campaign_Data.ipynb**
- **Steps**: Load data → Clean data types → Handle missing values → Outlier detection → Feature engineering → Train RandomForestRegressor → Predict sales
- **Problem**: Predict sales units from campaign data with messy, mixed-type features (currency symbols, percentages, missing values)
- **Key Features**:
  - Data cleaning: Remove currency symbols (£) and percentages (%)
  - Missing value imputation using median for numerical columns
  - Outlier detection using IQR method (identifies 2-3% outliers)
  - Feature engineering: Extract temporal features (Year, Month, DayOfWeek, Hour, IsWeekend)
  - Data preprocessing with StandardScaler and OneHotEncoder pipelines
  - RandomForestRegressor with hyperparameter tuning
  - Model evaluation metrics: MAE, RMSE, R²-score
- **Dataset**: 
  - Train: 8,000 records with Sales target
  - Test: 2,000 records without Sales (predictions only)
  - Features: Followers, EngagementRate (%), AdSpend (GBP), ContentQuality, Timestamp, Notes
- **Performance**: R² = 0.92 on training data
- **Domain**: Marketing Analytics & Sales Forecasting

### 8. **License Plate Detection & Blurring using YOLOv8**
- **Steps**: Load YOLOv8 model → Detect license plates in images/video → Extract detected regions → Apply Gaussian blur → Display/save results
- **Problem**: Automatically detect and blur license plates in images/videos for privacy protection
- **Key Features**:
  - YOLOv8 object detection model for accurate license plate localization
  - Preprocessing: Image normalization and resizing to model input specifications
  - Gaussian blur filter applied to detected license plate regions
  - Support for batch processing multiple images and video frames
  - Confidence threshold filtering for detection quality control
  - Bounding box visualization with confidence scores
  - Output generation: Blurred images/videos with detected regions
- **Dataset**: Real-world images containing vehicles with visible license plates
- **Technology Stack**: 
  - YOLOv8 (Ultralytics)
  - OpenCV for image/video processing
  - Python 3.8+
- **Use Cases**: Privacy protection in surveillance, dataset anonymization, automated video redaction
- **Domain**: Computer Vision & Privacy-Preserving ML

---

## 🛠️ Installation & Setup

```bash
# Clone repository
git clone https://github.com/Shivaprakashhv/AIMLProjects.git
cd AIMLProjects

# Install dependencies
pip install -r requirements.txt

# For specific projects
pip install torch transformers  # LoRA & RAG
pip install scikit-learn pandas  # ML projects
pip install ultralytics opencv-python  # YOLOv8
```


## 🚀 Getting Started

Each project folder contains:
- Jupyter notebooks with step-by-step implementations
- Detailed problem statements and solutions
- Dataset information and preprocessing steps
- Model evaluation metrics and performance results

---

## 👨‍💻 Author

**Shivaprakashhv**  
Passionate about AI/ML, Data Science, and building practical solutions.

---

*Last Updated: 2026-04-05*
