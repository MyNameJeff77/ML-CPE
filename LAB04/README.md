ชุดข้อมูลที่ใช้ Kaggle https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

### Objective

The objective of this lab is to learn how the K-Nearest Neighbor (KNN) algorithm works for classification problems. We compare different values of **k (3, 5, and 7)** to determine which model provides the highest accuracy on the Heart Disease dataset.

---

# Step 1 : Import Libraries

First, we import the libraries required for data analysis, visualization, preprocessing, model training, and evaluation.

- Pandas is used to load and manipulate data.
- Matplotlib and Seaborn are used to visualize data.
- Scikit-learn provides the KNN algorithm and evaluation tools.

---

# Step 2 : Load Dataset

Next, we load the Heart Disease dataset using **pandas.read_csv()**.

After loading the dataset, we display the first five rows using **head()** to verify that the data has been loaded correctly.

---

# Step 3 : Explore Dataset (EDA)

Before building a machine learning model, it is important to understand the dataset.

In this step we:

- Check the data type of each column using **info()**
- Display statistical information using **describe()**
- Check for missing values using **isnull().sum()**
- Count the number of samples in each target class

This helps us understand the dataset before training the model.

---

# Step 4 : Data Visualization

We visualize the target variable using a count plot.

This graph shows the number of patients with and without heart disease, making it easier to understand the class distribution.

---

# Step 5 : Feature Selection

Machine learning requires separating the dataset into:

- **X** = input features
- **y** = target variable

The **target** column is used as the prediction label, while all remaining columns become input features.

---

# Step 6 : Train-Test Split

The dataset is divided into two parts:

- 80% for training
- 20% for testing

The training set is used to train the model, while the testing set is used to evaluate its performance.

---

# Step 7 : Standardize Features

KNN calculates the distance between data points.

Because each feature has different units, we standardize the data using **StandardScaler()**.

Feature scaling prevents variables with large values from dominating the distance calculation.

---

# Step 8 : Train KNN Models

Three KNN models are created using different values of **k**.

- k = 3
- k = 5
- k = 7

Each model is trained using the training dataset.

After training, predictions are made on the testing dataset.

The accuracy of each model is then calculated.

---

# Step 9 : Compare Accuracy

The accuracy values of each model are collected into a table.

A line chart is also created to compare how different values of **k** affect model performance.

This makes it easy to identify the best-performing model.

---

# Step 10 : Find the Best k

The model with the highest accuracy is selected.

This value of **k** will be used for the final evaluation.

---

# Step 11 : Confusion Matrix

The best KNN model is evaluated using a Confusion Matrix.

This matrix compares the actual labels with the predicted labels and shows how many samples were classified correctly and incorrectly.

---

# Step 12 : Classification Report

The Classification Report displays:

- Precision
- Recall
- F1-score
- Accuracy

These metrics provide a more detailed evaluation of the model than accuracy alone.

---

# Step 13 : Conclusion

In this lab, the Heart Disease dataset was classified using the K-Nearest Neighbor algorithm.

The data was explored, preprocessed, standardized, and divided into training and testing sets.

Three different values of **k (3, 5, and 7)** were tested.

The model with the highest accuracy was selected as the best model.

The experiment shows that KNN is a simple but effective algorithm for classification problems when the data is properly preprocessed.