import numpy as np
import numpy.typing as npt
import pandas as pd
import copy
from sklearn import preprocessing  # type: ignore
from sklearn.impute import SimpleImputer  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import classification_report  # type: ignore
from sklearn.feature_selection import VarianceThreshold  #type: ignore


class SalaryPredictor:
    """
    A Logistic Regression Classifier used to predict someone's salary (from LONG ago)
    based upon their demographic characteristics like education level, age, etc. This
    task is turned into a binary-classification task with two labels:
      y = 0: The individual made less than or equal to 50k
      y = 1: The individual made more than 50k

    [!] You are free to choose whatever attributes needed to implement the SalaryPredictor;
    unlike the ToxicityFilter, there are no constraints of what you must include here.
    """

    def __init__(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """
        Creates a new SalaryPredictor trained on the given features from the
        preprocessed census data to predicted salary labels. Does so by:
        1. Preprocesses the training data
        2. Fits the Logistic Regression model to the transformed features
        3. Saves this model as an attribute for later use

        Parameters:
            X_train (pd.DataFrame):
                Pandas DataFrame consisting of the sample rows of attributes
                pertaining to each individual

            y_train (pd.DataFrame):
                Pandas DataFrame consisting of the sample rows of labels 
                pertaining to each person's salary
        """
        features = self.preprocess(X_train, True)

        self.lrbc = LogisticRegression(max_iter=5000)
        self.lrbc.fit(features, y_train)

        self.num_imputer
        self.cat_imputer
        self.encoder

    def preprocess(self, features: pd.DataFrame, training: bool = False) -> npt.NDArray:
        """
        Takes in the raw rows of individuals' characteristics to be used for
        salary classification and converts them into the numerical features that
        can be used both during training and classification by the LR model.

        Parameters:
            features [pd.DataFrame]:
                The data frame containing all inputs to be preprocessed where the
                rows are 1 per person to classify and the columns are their attributes
                that may require preprocessing, e.g., one-hot encoding the categorical
                attributes like education.

            training [bool]:
                Whether or not this preprocessing call is happening during training
                (i.e., in the SalaryPredictor's constructor) or during testing (i.e.,
                in the SalaryPredictor's classify method). If set to True, all preprocessing
                attributes like imputers and OneHotEncoders must be fit before transforming
                any features to numerical representations. If set to False, should NOT fit
                any preprocessors, and only use their transform methods.

        Returns:
            np.ndarray:
                Numpy Array composed of numerical features converted from the raw inputs.
        """
        # step one, clean up messy data
        features = features.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        # step two, replace missing data with a reasonable response
        # handle numbers and non-numbers separately since they will have different replacement methods
        num_columns = features.select_dtypes(include=["number"]).columns
        cat_columns = features.select_dtypes(include=["object"]).columns

        if training:
            # Fit imputers during training
            self.num_imputer = SimpleImputer(strategy="mean")  # replace ? with the mean when using numbers
            self.cat_imputer = SimpleImputer(strategy="most_frequent")  # replace ? with most frequent answer for non numbers
            # Fit and transform
            features[num_columns] = self.num_imputer.fit_transform(features[num_columns])
            features[cat_columns] = self.cat_imputer.fit_transform(features[cat_columns])

            # step three, change continuous variables into discrete ones
            # Fit OneHotEncoder during training
            self.encoder = preprocessing.OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoded_features = self.encoder.fit_transform(features[cat_columns])
            # Concatenate encoded categorical features with numerical features
            numerical_features = features[num_columns].to_numpy()
            final_features = np.hstack([numerical_features, encoded_features])

            # step four, standardization (so we don't hit our iteration limit)
            # Fit StandardScaler during training
            self.scaler = preprocessing.StandardScaler()
            final_features = self.scaler.fit_transform(final_features)

            # step five, feature selection
            self.selector = VarianceThreshold(threshold=0.01)
            final_features = self.selector.fit_transform(final_features)

        else:  # time for testing
            # Use previously fitted imputers
            features[num_columns] = self.num_imputer.transform(features[num_columns])
            features[cat_columns] = self.cat_imputer.transform(features[cat_columns])
            # Use the already-fitted encoder
            encoded_features = self.encoder.transform(features[cat_columns])
            # Concatenate encoded categorical features with numerical features
            numerical_features = features[num_columns].to_numpy()
            final_features = np.hstack([numerical_features, encoded_features])

            # step four, standardization (so we don't hit our iteration limit)
            # Use the already-fitted scaler
            final_features = self.scaler.transform(final_features)

            # step five, feature selection
            final_features = self.selector.transform(final_features)

        return final_features

    def classify(self, X_test: pd.DataFrame) -> list[int]:
        """
        Takes as input a data frame containing input user demographics, uses the predictor's
        preprocessing to transform these into the ndarray of numerical features, and then
        returns a list of salary classifications, one for each individual.

        [!] Note: Should use the preprocess method with training parameter set to False!

        Parameters:
            X_test (list[str]):
                A data frame where each row is a new individual with characteristics like
                age, education, etc. that the salary predictor must assess.

        Returns:
            list[int]:
                A list of classifications, one for each individual, where the
                index of the output class corresponds to the index of input person.
                The ints represent the classes such that y=0: <=50k and y=1: >50k
        """
        return list(self.lrbc.predict(self.preprocess(X_test, False)))

    def test_model(self, X_test: "pd.DataFrame", y_test: "pd.DataFrame") -> tuple[str, dict]:
        """
        Takes the test-set as input (2 DataFrames consisting of test inputs
        and their associated labels), classifies each, and then prints
        the classification_report on the expected vs. given labels.

        Parameters:
            X_test [pd.DataFrame]:
                Pandas DataFrame consisting of the test rows of individuals

            y_test [pd.DataFrame]:
                Pandas DataFrame consisting of the test rows of labels pertaining 
                to each individual

        Returns:
            tuple[str, dict]:
                Returns the classification report in two formats as a tuple:
                [0] = The classification report as a prettified string table
                [1] = The classification report in dictionary format
                In either format, contains information on the accuracy of the
                classifier on the test data.
        """
        prediction = self.classify(X_test)
        return (classification_report(y_test, prediction, output_dict=False),
                classification_report(y_test, prediction, output_dict=True))
