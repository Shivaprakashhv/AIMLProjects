# AI ML Projects

## 📁 Project Files & Solutions

### 1. **LoRA Fine Tuning.py**
- **Steps**: Load model → Configure LoRA → Prepare dataset → Train → Save adapter
- **Problem**: Efficiently fine-tune large LLMs with minimal memory using parameter-efficient LoRA

### 2. **RAG_Pipeline.py**
- **Steps**: Load data → Chunk documents → Generate embeddings → Store in Milvus → Retrieve & Generate answers
- **Problem**: Build semantic search + LLM system to answer questions from documents

### 3. **Click_Through_Rate_Prediction.ipynb**
- **Steps**: EDA → Handle imbalance → Extract temporal features → Train model → Predict clicks
- **Problem**: Predict ad clicks in highly imbalanced dataset (93% negative class)

### 4. **Amazon_Product_Recommendation.ipynb**
- **Steps**: Load data → EDA → Extract features → Build recommendation model
- **Problem**: Build product recommendation system from Amazon reviews & ratings data

### 5. **IPL_Data_Analysis.ipynb**
- **Steps**: Download data → Clean team names → EDA → Extract patterns → Visualize trends
- **Problem**: Analyze cricket performance metrics across IPL seasons (2008-2024)

### 6. **E_commerce_Marketing_and_sales.ipynb**
- **Steps**: Analyze acquisitions → Monthly trends → Retention cohort analysis
- **Problem**: Identify seasonal patterns and improve customer retention strategies

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

### 8. **License Plate Detection & Blurring using YOLOv8**
