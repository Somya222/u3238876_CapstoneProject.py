import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


class NbaDataAnalysis:
    def __init__(self, nba_eda_df):
        self.nba_eda_df = nba_eda_df

    def nba_data_cleaning(self):
        st.title("Analyzing the NBA Elo ratings Dataset: An Exploration Through Visualizations")

        st.write('This notebook analyses the elo ratings of the NBA for each team.'
                 'Based on the reatings it shows the distrubution of the elo ratings on each team, therefor analyse '
                 'the average ratings')

        # check the data set
        st.subheader('First 10 rows:')
        st.write(self.nba_eda_df.head(10))
        st.write("\n")

        st.subheader('Last 5 rows:')
        st.write(self.nba_eda_df.tail())
        st.write("\n")

        # check the data types of the columns
        st.subheader('Data Types:')
        st.write(self.nba_eda_df.dtypes)
        st.write("\n")

        # check for missing values
        st.subheader('Missing Values:')
        st.write(self.nba_eda_df.isnull().sum())
        st.write("We have missing values for neutral carm-elo1_pre, raptor_prob1, importance, ratings")
        st.write("\n")

        # calculate the percentage of missing values in each column
        missing_values = (self.nba_eda_df.isna().sum() / len(self.nba_eda_df)) * 100
        st.write('Percentage of Missing Values in Each Column:')

        st.write("neutral, playoff columns has no values at all. So, we will remove these."
                 "For raptor1_pre, raptor2_pre, raptor_prob1,raptor_prob2 I am un certain what these data defnies to."
                 "Hence I am removeing it as they are not required for the analysis."
                 "quality, impiortance has the missing values.")
        st.write("\n")

        # Plotting the missing values [1]
        fig = plt.figure(figsize=(10, 7))
        plt.bar(x=missing_values.index, height=missing_values.values)
        plt.xticks(rotation=90)
        plt.xlabel('Columns')
        plt.ylabel('Percentage of Missing Values')
        plt.title('Percentage of Missing Values in Nba Dataset')
        st.pyplot(fig)

        st.write("For this analysis plotting a graph to display the missing values from the dataset")

        st.write("\n")

        # Drop the columns with missing values
        self.new_nba_df = self.nba_eda_df.drop(['neutral', 'playoff',
                                                'raptor1_pre', 'raptor2_pre',
                                                'raptor_prob1', 'raptor_prob2', 'carm-elo1_pre', 'carm-elo2_pre',
                                                'carm-elo_prob1', 'carm-elo_prob2', 'carm-elo1_post',
                                                'carm-elo2_post', 'quality', 'importance'], axis=1)

        # Handing NA value by using median values
        elo1_post_median = self.new_nba_df['elo1_post'].median()
        elo2_post_median = self.new_nba_df['elo2_post'].median()
        score1_median = self.new_nba_df['score1'].median()
        score2_median = self.new_nba_df['score2'].median()
        self.new_nba_df['score2'].fillna(score2_median, inplace=True)
        self.new_nba_df['score1'].fillna(score1_median, inplace=True)
        self.new_nba_df['elo2_post'].fillna(elo2_post_median, inplace=True)
        self.new_nba_df['elo1_post'].fillna(elo1_post_median, inplace=True)

        # check the data types of the columns
        st.subheader('Dataset info after handling missing values:')
        st.write(self.nba_eda_df.isnull().sum())
        st.write("\n")

        # st.write("Add New Column")

        # Create a new column 'winning_team' based on the team that scored higher
        # If team1 has a higher score, set winning_team to team1
        # Otherwise, set winning_team to team2

        self.new_nba_df['winning_team'] = self.new_nba_df.apply(lambda row: row['team1']
        if row['score1'] > row['score2']
        else row['team2'], axis=1)
        st.write("This data set of Nba elo ratings did not provide any columns"
                 "with winning team fo I am creating a new column for the winning team.")

        # Check for duplicate rows
        st.write("\n")
        st.subheader("Checking Duplicate")
        duplicates = self.new_nba_df.duplicated()
        st.write("\n")

        # display the summary statistics of the data
        st.subheader('Summary Statistics:')
        st.write(self.new_nba_df.describe())
        st.write("\n")

        # Question 1
        st.subheader("Question 1 : Calculate winning frequency for each team?")

        # Calculate winning frequency for each team
        winning_freq = self.new_nba_df['winning_team'].value_counts(normalize=True)

        # Create a bar plot
        fig = plt.figure(figsize=(10, 6))
        plt.bar(winning_freq.index, winning_freq.values)
        plt.title('Winning Frequency for each team')
        plt.xlabel('Team')
        plt.ylabel('Winning Frequency')
        plt.xticks(rotation=45)
        fig.tight_layout()
        st.pyplot(fig)

        st.write("It depicts the winning frequency calculated for each team based on the scores")
        st.write("\n")

        # Question 2
        st.subheader("Question 2: What is the distribution of elo_prob1 for all games in the dataset?")
        fig = plt.figure(figsize=(10, 6))
        plt.hist(self.new_nba_df['elo_prob1'], bins=30)
        plt.title('Distribution of elo_prob1')
        plt.xlabel('elo_prob1')
        plt.ylabel('Count')
        st.pyplot(fig)
        st.write("Team 1's probability of winning based on Elo rating")
        st.write("\n")

        # Question 3
        st.subheader("Question 03: What is the distribution of elo_prob2 for all games in the dataset?")

        fig = plt.figure(figsize=(10, 6))
        plt.hist(self.new_nba_df['elo_prob2'], bins=30)
        plt.title('The distribution of elo_prob2 for all games')
        plt.xlabel('elo_prob2')
        plt.ylabel('Count')
        st.pyplot(fig)
        st.write("There is a significance change in the probability after the match compare to elo 1")
        st.write("\n")

        # Question 4
        st.subheader("Question 4: Which team has the highest average elo1_pre rating?")

        # across all games in the dataset?
        team_elo = self.new_nba_df.groupby('team1')['elo1_pre'].mean()
        # Sort the resulting Series in descending order
        team_elo_sorted = team_elo.sort_values(ascending=False)
        # Print the team with the highest average elo1_pre rating
        st.write("Team with the highest average elo1_pre rating:", team_elo_sorted.index[0])
        team_elo = self.new_nba_df.groupby('team1')['elo1_pre'].mean()
        # Find the team with the highest average elo1_pre rating
        highest_avg_elo_team = team_elo.idxmax()
        highest_avg_elo = team_elo.max()

        st.write(
            f"The team with the highest average elo1_pre rating is {highest_avg_elo_team}"
            f" with an average rating of {highest_avg_elo}.")

        # Visualize the results
        fig = plt.figure(figsize=(10, 6))
        team_elo.plot(kind='bar', figsize=(12, 6), title='Average elo1_pre rating by team')
        plt.xlabel('Team')
        plt.ylabel('Average elo1_pre rating')
        st.pyplot(fig)

        st.write("\n")

        # Question 5
        st.subheader("Question 5: Is there a correlation between the difference in elo1_pre?")
        # "Question 5 : Is there a correlation between the difference in elo1_pre?")
        # ratings between two teams and the final score difference?
        self.new_nba_df["total_rating"] = self.new_nba_df["elo1_pre"] - self.new_nba_df["elo2_pre"]
        # Create a scatter plot of elo_diff vs score_diff
        sns.scatterplot(data=self.new_nba_df, x="total_rating", y="score1")
        sns.scatterplot(data=self.new_nba_df, x="total_rating", y="score2")
        # Calculate the correlation coefficient
        correlation = self.new_nba_df["total_rating"].corr(self.new_nba_df["score1"])
        correlation = self.new_nba_df["total_rating"].corr(self.new_nba_df["score2"])
        st.write("Correlation coefficient: ", correlation)
        # Create the confusion matrix
        cm = np.array([[0, 1000], [1000, 0]])

        # Plot the confusion matrix
        fig = plt.figure(figsize=(10, 6))
        plt.imshow(cm, cmap='Oranges')
        plt.title('Correlation between the difference in elo1_pre')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.colorbar()
        st.pyplot(fig)

        st.write("\n")

        st.subheader("Part 2 : Model Building")

    def predictive_model(self, fig=None):
        # Select the relevant features and target variable
        features = ['elo1_pre', 'elo2_pre', 'elo_prob1', 'elo_prob2']
        target = 'winning_team'

        # Split the data into training and testing sets
        X = self.new_nba_df[features]
        y = self.new_nba_df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=75)

        # Create Random Forest classifier
        self.rfc = RandomForestClassifier()
        self.rfc.fit(X_train, y_train)

        # Create Decision Tree classifier
        self.dtc = DecisionTreeClassifier()
        self.dtc.fit(X_train, y_train)

        # Create SVM classifier
        self.svm = SVC(kernel='linear')
        self.svm.fit(X_train, y_train)

        # Evaluate classifiers on test set
        st.write("Random Forest accuracy:", self.rfc.score(X_test, y_test))
        st.write("Decision Tree accuracy:", self.dtc.score(X_test, y_test))
        st.write("SVM accuracy:", self.svm.score(X_test, y_test))

        # Build confusion matrix for Random Forest classifier

        rfc_pred = self.rfc.predict(X_test)
        rfc_cm = confusion_matrix(y_test, rfc_pred)
        fig = plt.figure(figsize=(10, 7))
        sns.heatmap(rfc_cm, annot=True, cmap='Blues')
        plt.title("Random Forest Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        st.pyplot(fig)
        st.write("\n")

        # Plot confusion matrix for Decision Tree classifier

        dtc_pred = self.dtc.predict(X_test)
        dtc_cm = confusion_matrix(y_test, dtc_pred)
        fig = plt.figure(figsize=(10, 7))
        sns.heatmap(dtc_cm, annot=True, cmap='Blues')
        plt.title("Decision Tree Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        st.pyplot(fig)
        st.write("\n")

        # Plot confusion matrix for SVM classifier

        svm_pred = self.svm.predict(X_test)
        svm_cm = confusion_matrix(y_test, svm_pred)
        fig = plt.figure(figsize=(10, 7))
        sns.heatmap(svm_cm, annot=True, cmap='Blues')
        plt.title("SVM Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        st.pyplot(fig)
        st.write("\n")

        st.write(
            "We can see that Random Forest classifier has the highest accuracy scores among all but that is still low."
            "Lest do some hyper tunning to immprove the accuracy on best model")
        st.write("\n")
        st.write("Hypertunning")

        # Define the parameter grid to search over
        param_grid = {
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [5, 10, 20, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Create a random forest classifier
        rf_h = RandomForestClassifier()

        # Use GridSearchCV to search over the parameter grid
        grid_search = GridSearchCV(rf_h, param_grid, cv=5, n_jobs=-1)

        # Fit the grid search to the data
        grid_search.fit(X_train, y_train)

        # Print the best parameters and best score
        st.write(f"Best parameters: {grid_search.best_params_}")
        st.write(f"Best score: {grid_search.best_score_}")

        # Define the best parameters from the grid search
        best_params = grid_search.best_params_
        # Create a random forest classifier with the best parameters
        rf_best = RandomForestClassifier(random_state=20, **best_params)
        # Fit the classifier to the training data
        rf_best.fit(X_train, y_train)
        # Evaluate the classifier on the test data
        st.write("Random Forest accuracy (with best params):", rf_best.score(X_test, y_test))

        st.write("Score still low aftr hyper tunning."
                 "Seems like this data set is no t good enough to predict the winning team. Probably,"
                 "adding more varaible in the data set could improve the model accuracy")


if __name__ == '__main__':
    nba_elo_df = pd.read_csv('nba_elo_latest.csv')
    nba_data_analysis_df = NbaDataAnalysis(nba_elo_df)
    nba_data_analysis_df.nba_data_cleaning()
    nba_data_analysis_df.predictive_model()
