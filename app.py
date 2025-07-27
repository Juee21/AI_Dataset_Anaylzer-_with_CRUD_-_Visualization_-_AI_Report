import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai
import os
from dotenv import load_dotenv

# 🔑 Set your Gemini API Key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('models/gemini-2.0-flash')

# 🚀 Create necessary folders
os.makedirs("datasets", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# 🎯 Streamlit setup
st.set_page_config(page_title="AI Dataset Analyzer", layout="wide")
st.title("📊 AI Dataset Analyzer with CRUD + Visualization + AI Reports")

# 🚩 Sidebar menu
menu = st.sidebar.radio(
    "Menu",
    ["Upload Dataset", "View/Edit Dataset", "Visualization", "AI Report", "AI Query"]
)

# ✅ Session variables
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.dataset_name = None

# ================================================
# 📥 Upload Dataset
# ================================================
if menu == "Upload Dataset":
    st.subheader("Upload a CSV Dataset")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    dataset_name = st.text_input("Dataset name (no spaces):")

    if uploaded_file and dataset_name:
        save_path = os.path.join("datasets", dataset_name + ".csv")
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ Dataset '{dataset_name}' saved successfully!")

    st.subheader("Available Datasets")
    datasets = os.listdir("datasets")

    if datasets:
        selected_dataset = st.selectbox("Select dataset to load", datasets)

        if st.button("Load Dataset"):
            df = pd.read_csv(os.path.join("datasets", selected_dataset))
            st.session_state.df = df
            st.session_state.dataset_name = selected_dataset
            st.success(f"✅ Loaded '{selected_dataset}' successfully!")
            st.write(df.head())

        if st.button("Delete Selected Dataset"):
            os.remove(os.path.join("datasets", selected_dataset))
            st.success(f"🗑️ Deleted dataset '{selected_dataset}'")
    else:
        st.info("No datasets available. Please upload one.")

# ================================================
# 👀 View / Edit Dataset
# ================================================
if menu == "View/Edit Dataset":
    if st.session_state.df is not None:
        st.subheader(f"View & Edit Dataset: {st.session_state.dataset_name}")

        df = st.session_state.df.copy()

        edited_df = st.data_editor(df, num_rows="dynamic")
        st.session_state.df = edited_df

        st.success("✅ Changes saved in session.")

        # Download modified CSV
        csv = edited_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Modified Dataset as CSV",
            data=csv,
            file_name=st.session_state.dataset_name,
            mime='text/csv',
        )
    else:
        st.warning("⚠️ Please upload and load a dataset first.")

# ================================================
# 📊 Visualization
# ================================================
if menu == "Visualization":
    if st.session_state.df is not None:
        df = st.session_state.df

        st.subheader("Data Visualization")

        chart_type = st.selectbox(
            "Select Chart Type",
            ["Histogram", "Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart"]
        )

        columns = df.columns.tolist()

        x = st.selectbox("X-axis", columns)
        y = st.selectbox("Y-axis (optional for some charts)", ["None"] + columns)

        fig, ax = plt.subplots()

        try:
            if chart_type == "Histogram":
                ax.hist(df[x].dropna(), bins=20)
                ax.set_xlabel(x)
                ax.set_title(f"Histogram of {x}")

            elif chart_type == "Bar Chart":
                counts = df[x].value_counts()
                ax.bar(counts.index.astype(str), counts.values)
                ax.set_title(f"Bar Chart of {x}")

            elif chart_type == "Line Chart":
                if y != "None":
                    ax.plot(df[x], df[y])
                    ax.set_xlabel(x)
                    ax.set_ylabel(y)
                    ax.set_title(f"Line Chart: {y} vs {x}")
                else:
                    st.error("Y-axis is required for Line Chart.")

            elif chart_type == "Scatter Plot":
                if y != "None":
                    ax.scatter(df[x], df[y])
                    ax.set_xlabel(x)
                    ax.set_ylabel(y)
                    ax.set_title(f"Scatter Plot: {y} vs {x}")
                else:
                    st.error("Y-axis is required for Scatter Plot.")

            elif chart_type == "Pie Chart":
                counts = df[x].value_counts()
                ax.pie(counts, labels=counts.index.astype(str), autopct='%1.1f%%')
                ax.set_title(f"Pie Chart of {x}")

            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error generating plot: {e}")

    else:
        st.warning("⚠️ Please upload and load a dataset first.")

# ================================================
# 📑 AI Report Generator
# ================================================
if menu == "AI Report":
    if st.session_state.df is not None:
        df = st.session_state.df

        st.subheader("🧠 AI-Generated Data Report")

        stats = df.describe(include='all').to_string()
        missing = df.isnull().sum().to_string()

        prompt = f"""
You are an expert data analyst. Write a detailed report for the following dataset.

Statistics:
{stats}

Missing values:
{missing}

Summarize insights, trends, anomalies, and potential actions.
"""

        if st.button("Generate AI Report"):
            with st.spinner("🧠 Generating report..."):
                response = model.generate_content(prompt)
                report = response.text
                st.text_area("📄 AI Report", value=report, height=400)

                # Save report as TXT
                report_file = os.path.join(
                    "reports",
                    st.session_state.dataset_name.replace(".csv", "_report.txt")
                )
                with open(report_file, "w") as f:
                    f.write(report)

                with open(report_file, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Report as TXT",
                        data=f,
                        file_name=os.path.basename(report_file),
                        mime='text/plain',
                    )
    else:
        st.warning("⚠️ Please upload and load a dataset first.")

# ================================================
# 💡 AI Query (Natural Language)
# ================================================
if menu == "AI Query":
    if st.session_state.df is not None:
        st.subheader("💡 Ask AI about your Dataset")
        df = st.session_state.df

        # Improved prompt with strict formatting rules
        prompt_template = """
        You are a data analyst. Given this DataFrame with columns: {columns}
        The user asked: "{question}"

        Generate VALID Python code that:
        1. Must be error-free when executed with: df, pd, np, plt
        2. Stores results in 'result' (for tables) or uses plt (for visualizations)
        3. NEVER includes explanations or markdown
        4. Follows this exact format:
        ```python
        # CODE HERE (ONLY PANDAS/MATPLOTLIB)
        ```

        Example response for "Show average salary by department":
        ```python
        result = df.groupby('department')['salary'].mean()
        ```
        """

        question = st.text_input("Ask your question")
        
        if question and st.button("Submit"):
            with st.spinner("Analyzing..."):
                try:
                    # Generate code with strict formatting
                    response = model.generate_content(
                        prompt_template.format(columns=', '.join(df.columns), question=question)
                    )
                    
                    # Extract code from markdown blocks (if present)
                    code = response.text
                    if '```python' in code:
                        code = code.split('```python')[1].split('```')[0]
                    
                    # Validate code before execution
                    if not code.strip():
                        raise ValueError("No code generated")
                    
                    # Prepare execution environment
                    env = {
                        'df': df,
                        'pd': pd,
                        'np': np,
                        'plt': plt,
                        'result': None
                    }
                    
                    # Execute
                    exec(code, {'__builtins__': None}, env)
                    
                    # Display results
                    if env['result'] is not None:
                        st.dataframe(env['result'])  # For tables
                    elif plt.gcf().get_axes():
                        st.pyplot(plt.gcf())  # For plots
                    else:
                        st.info("No results returned. Try being more specific.")
                        
                except SyntaxError as e:
                    st.error(f"Invalid code generated. Please try rephrasing your question.")
                    st.code(f"Error: {e}\nGenerated code:\n{code}", language='python')
                except Exception as e:
                    st.error(f"Couldn't process your question: {str(e)}")
                    st.info("Try questions like:\n- 'Show sales by region'\n- 'Plot age distribution'")
    else:
        st.warning("Please load a dataset first.")