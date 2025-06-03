import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Load the dataset
df = pd.read_csv('meesho_reviews_extended_dataset.csv')
print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())


# Check for duplicates
print(df.duplicated().sum())
# Remove duplicates
df = df.drop_duplicates()
print(df.duplicated().sum())


df['complaint_reason']=df['complaint_reason'].fillna(0)
print(df['complaint_reason'].isnull().sum())
print(df.isnull().sum())

df.to_csv ("cleaned_meesho_reviews.csv", index=False)
print(df)

#Verify if missing values are filled
missing_values_after = df.isnull().sum()
print("Missing values after filling:")
print(missing_values_after)



#Real-Time Customer Sentiment Analysis for Product Reviews 
#Classify and analyze live product reviews to gauge customer satisfaction instantly.
#EDA analysis
#Univariate analysis
#Visualize the distribution of ratings
plt.figure(figsize=(8, 6))
sns.countplot(x='Rating', data=df, palette='viridis')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.legend()
plt.show()

#Bivariate analysis
#Visualize the relationship between ratings and review text 
df['Review_text'] = df['Review_text'].apply(lambda x: len(x.split()))
plt.figure(figsize=(8, 6))
sns.boxplot(x='Review_text', y='Rating', data=df, palette='viridis')
plt.title('Ratings vs Review Length')
plt.xlabel('Review_text')
plt.ylabel('Rating')
plt.tight_layout()
plt.show()


#Multivariate analysis
#Visualize the relationship between ratings, review text, and complaint reason
plt.figure(figsize=(10, 6))
sns.violinplot(x='Review_text', y='Rating', hue='complaint_reason', data=df, palette='viridis')
plt.title('Ratings vs Review Length and Complaint Reason')
plt.xlabel('Review_text')
plt.ylabel('Rating')
plt.tight_layout()
plt.legend()
plt.show()




#Correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr(numeric_only=True)  # Ensure only numeric columns are included
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.legend()
plt.show()



#Linear Reggression
# Prepare the data for linear regression
X= df[['previous_purchases']]  
y = df['Rating']                

# Step 2: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Predict
y_pred = model.predict(X_test)

# Step 5: Show results
print(f"Predicted Rating for test data: {y_pred}")

#Step6
# Evaluate the model
# Calculate the mean squared error and R-squared value
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Plot regression line
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel("Previous Purchases")
plt.ylabel("Rating")
plt.legend()
plt.title("Simple Linear Regression")
plt.show()

#plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual vs Predicted Ratings')
plt.show()


# Predict the rating for a new customer with a given number of previous purchases
new_data = pd.DataFrame({'previous_purchases': [5]})  # Example: 5 previous purchases
predicted_rating = model.predict(new_data)
print(f'Predicted Rating for new customer: {predicted_rating[0]}')






















