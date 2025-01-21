import streamlit as st
import openai
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt

load_dotenv()

# Securely retrieve the OpenAI API key
openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# Check if the API key is set
if not openai.api_key:
    st.error("OpenAI API Key is missing. Please check your Streamlit secrets or .env file.")
    st.stop()


# Initialize session state:
if 'messages' not in st.session_state:
    st.session_state.messages = []

st.title("Financial Data Q&A with Historical and Predictive Analysis")

st.sidebar.header("Upload your CSV file")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

required_columns = [
    "Year",
    "Market cap ($B)",
    "Revenue ($B)",
    "Earnings ($B)",
    "Operating Margin (%)",
    "Shares Outstanding ($B)",
    "Total assets ($B)"
]

if uploaded_file:
    try:
        # Load the entire CSV content
        df = pd.read_csv(uploaded_file)

        # Validate if the required columns are present
        if all(col in df.columns for col in required_columns):
            st.write("CSV Data Preview:", df)

            # Prepare data for regression
            df = df.sort_values(by="Year")  # Ensure data is sorted by year

            X = df[['Year']].values
            y = df['Revenue ($B)'].values

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train a Linear Regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict future revenue for the next 10 years
            future_years = np.arange(df['Year'].max() + 1, df['Year'].max() + 11).reshape(-1, 1)
            future_revenue = model.predict(future_years)

            # Create DataFrame for future predictions
            future_df = pd.DataFrame({
                'Year': future_years.flatten(),
                'Predicted Revenue ($B)': future_revenue
            })

            # Function to retrieve data based on the question
            def retrieve_data(question):
                """
                Determines whether the question is about historical or predictive data.
                """
                if "predict" in question.lower() or "forecast" in question.lower() or "future" in question.lower():
                    # Predictive response
                    return f"Predicted revenue for the next 10 years:\n{future_df.to_string(index=False)}"
                else:
                    # Historical data response
                    return df.to_string(index=False)

            # Display chat history
            if st.session_state.messages:
                for message in st.session_state.messages:
                    role, content = message["role"], message["content"]
                    if role == "user":
                        st.write(f"**User:** {content}")
                    else:
                        st.write(f"**Assistant:** {content}")

            question = st.text_area("Ask a question about the financial data:")

            if st.button("Get Answer"):
                # Add user question to chat history
                st.session_state.messages.append({"role": "user", "content": question})

                # Retrieve relevant data
                relevant_data = retrieve_data(question)

                # Use OpenAI ChatCompletion to generate an answer with RAG approach
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": """
                        Instructions:
                        1. Use the dataset to answer questions accurately.
                        2. For predictive questions (e.g., forecast, future, predict), use the predictive data.
                        3. Summarize or conclude answers to avoid incomplete responses.
                        4. Provide responses in the requested format (e.g., currency, percentage, etc.).
                        5. Do not answer questions unrelated to the financial data.
                        """},
                        {"role": "user",
                         "content": f"The following is financial data in tabular format:\n{relevant_data}\n\nAnswer the question: {question}"}
                    ],
                    max_tokens=300
                )

                """Extract the assistant's answer"""
                answer = response['choices'][0]['message']['content'].strip()

                """Add assistant response to chat history"""
                st.session_state.messages.append({"role": "assistant", "content": answer})

                """Display the assistant's answer"""
                st.write("**Assistant:**", answer)

            """Display predictive analysis results"""
            st.header("Predictive Analysis: Revenue Forecast")
            st.write("Future Revenue Predictions (Next 10 Years):")
            st.dataframe(future_df)

            """Plot historical and predicted revenue"""
            plt.figure(figsize=(10, 6))
            plt.plot(df['Year'], df['Revenue ($B)'], label="Historical Revenue", marker='o')
            plt.plot(future_df['Year'], future_df['Predicted Revenue ($B)'], label="Predicted Revenue", marker='x', linestyle='--')
            plt.xlabel("Year")
            plt.ylabel("Revenue ($B)")
            plt.title("Revenue Forecast")
            plt.legend()
            st.pyplot(plt)

        else:
            missing_cols = set(required_columns) - set(df.columns)
            st.error(f"The uploaded CSV is missing required columns: {', '.join(missing_cols)}. Please upload a valid file.")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload a CSV file to get started.")
