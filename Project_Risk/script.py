import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def encodage(df):
    code = {"AAA" : 10,
            "AA" : 9,
            "A" : 8,
            "BBB" : 7,
            "BB" : 6,
            "B" : 5,
            "CCC" : 4,
            "CC" : 3,
            "C" : 2}
    if "Credit Rating" in df.columns:
              df["Credit Rating"] = df["Credit Rating"].map(code)    
    return df

def features(df):
    df['Profitability_Ratio'] = pd.to_numeric(df['Profit']) / pd.to_numeric(df['Revenue'])
    df["Profit_per_employee"] = df["Profit"]/df["Employee Count"]
    df["RnD_Spend_to_revenue"] = df["Research and Development Spend"]/pd.to_numeric(df["Revenue"])
    df["Earnings_to_Market_Cap"] = df["Profit"]/df["Market Capitalization"]
    return df 

def processing(df):
    df = features(df)
    df = encodage(df)
    X = df.drop("Risk", axis=1)
    y = df["Risk"]
    return X, y

def make_predictions(input_data, model, X_train, y_train, encoder=None, scaler=None):
    input_data_copy = input_data.copy()
    if encoder:
        input_data_copy = encoder(input_data_copy)
    feature_names = X_train.columns.tolist()
    X_input = input_data_copy[feature_names]
    if scaler:
        X_input = scaler.transform(X_input)
    predictions = model.predict(X_input)
    input_data_copy["Risk"] = predictions
    return input_data_copy

def main():
    # Creation of the ArgumentParser object
    parser = argparse.ArgumentParser(description='Predict risk for test dataset')

    # Add output argument (--output)
    parser.add_argument('--output', dest='output_file', type=str, required=True,
                        help='Output file for predicted labels')
    
    # Parsing command line arguments
    args = parser.parse_args()

    #Data import
    training_data = pd.read_csv("training_dataset.csv")

    # Encodage and processing
    X, y = processing(training_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    # Model ML
    rf_model = RandomForestClassifier(bootstrap=True,
                                      class_weight={0:1, 1:3.5},
                                      max_depth=5,
                                      min_samples_leaf=1,
                                      min_samples_split=10,
                                      n_estimators=75)
    rf_model.fit(X_train_scaled, y_train)
    y_pred = rf_model.predict(X_test_scaled)
    classification_rep = classification_report(y_test, y_pred)
    print("Classifaction report :\n", classification_rep)

    # Predictions 
    final_test_data = pd.read_csv("final_test.csv")
    final_test_data = features(final_test_data)
    final_test_data = encodage(final_test_data)
    predictions = make_predictions(final_test_data, rf_model, X_train, y_train, scaler=scaler)

    # Save predictions to a specified CSV file
    output_file_path = args.output_file
    predictions[["Company ID", "Risk"]].to_csv(output_file_path, index=False)
    print(f"Predictions saved to {output_file_path}")    

if __name__ == '__main__':
    main()