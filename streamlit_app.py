import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Function to train the model and make predictions
def train_and_predict(csv_file, feature1, feature2):
    # Read the CSV file
    music_data = pd.read_csv(csv_file)
    
    # Assuming 'genre' is the target variable
    X = music_data.drop(columns=['genre'])
    y = music_data['genre']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train the Decision Tree model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Make predictions for the provided features

    predictions = model.predict([np.array([feature1, feature2])])

    return predictions,accuracy_score(y_test,model.predict(X_test))

# Streamlit app
def main():
    st.title("Music Genre Prediction App")
    # Collecting "yes" or "no" using st.checkbox
    user_response = st.checkbox("Check if want to upload the custom CSV data set ")
    
    # Display the user's response
    if user_response:

        #display of sample csv file
        if st.checkbox("Check to show samle CSV File(.csv)"):
            st.image("image.png", caption="samle csv file image", use_column_width=True)

        # File uploader for CSV file
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        
        if uploaded_file is not None:
            # Display uploaded file
            st.write("Uploaded CSV file:")
            st.write(uploaded_file)
    
            # Get user inputs for feature1 and feature2
            feature1 = st.text_input("Enter value for Age:")
            feature2 = st.text_input("Enter gender ('0' for female and '1' for male):")
    
            # Make predictions when user clicks the button
            if st.button("Make Predictions"):

                if int(feature1)>=0 and int(feature2)>=0 and int(feature2)<=1:
   
                    try:
                        predictions,score = train_and_predict(uploaded_file, float(feature1), float(feature2))
                        st.markdown(f'<p style="color:red;">Predicted Genre : {predictions[0]}</p>', unsafe_allow_html=True)
                        accuracy_percentage = score * 100
                        st.write(f"Accuracy of the model: {accuracy_percentage:.2f}%")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.write("Invalid feature's value")
    else:
        # Get user inputs for feature1 and feature2
        feature1 = st.text_input("Enter value for Age:")
        feature2 = st.text_input("Enter gender ('0' for female and '1' for male):")

        # Make predictions when user clicks the button
        if st.button("Make Predictions"):
            if int(feature1)>=0 and int(feature2)>=0 and int(feature2)<=1:
                try:
                    predictions, score = train_and_predict( 'music1.csv', float(feature1), float(feature2))
                    st.markdown(f'<p style="color:red; font-weight: bold;">Predicted Genre : {predictions[0]}</p>', unsafe_allow_html=True)
                    accuracy_percentage = score * 100
                    st.write(f"Accuracy of the model: {accuracy_percentage:.2f}%")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.write("Invalid feature's value")

if __name__ == "__main__":
    main()
