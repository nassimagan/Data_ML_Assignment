import time
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.training.train_pipeline import TrainingPipeline
from src.constants import CM_PLOT_PATH, LABELS_MAP, SAMPLES_PATH,RAW_DATASET_PATH
from PIL import Image

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

def main():
    st.title("Resume Classification Dashboard")
    sidebar_options = st.sidebar.selectbox(
        "Dashboard Modes",
        ("EDA", "Training", "Inference")
    )

    if sidebar_options == "EDA":
        exploratory_data_analysis()
    elif sidebar_options == "Training":
        training_pipeline()
    else:
        resume_inference()

def exploratory_data_analysis():
    st.header("Exploratory Data Analysis")
    st.info("In this section,we created insightful graphs "
            "about the resume dataset.")
    df = pd.read_csv(RAW_DATASET_PATH)

    if df is None:
        st.warning("Please load your dataset to perform exploratory data analysis.")
        return

        # Summary statistics
    st.subheader("Statistical Descriptions")
    st.write("Summary statistics of the dataset:")
    st.write(df.describe())



def training_pipeline():
    st.header("Pipeline Training")

    name = st.text_input('Pipeline name', placeholder='Naive Bayes')
    serialize = st.checkbox('Save pipeline')
    train = st.button('Train pipeline')

    if train:
        train_pipeline(name, serialize)

def train_pipeline(model_name, serialize):
    with st.spinner('Training pipeline, please wait...'):
        try:
            tp = TrainingPipeline()
            tp.train(serialize=serialize, model_name=model_name)
            accuracy, f1 = tp.get_model_performance()
            display_metrics(accuracy, f1)
            tp.render_confusion_matrix()
            
        except Exception as e:
            display_error_message('Failed to train the pipeline!', e)


def display_metrics(accuracy, f1):
    col1, col2 = st.columns(2)
    col1.metric(label="Accuracy score", value=f"{round(accuracy, 4)}")
    col2.metric(label="F1 score", value=f"{round(f1, 4)}")
    st.image(Image.open(CM_PLOT_PATH), width=850)

def display_error_message(message, exception):
    st.error(message)
    st.exception(exception)

def resume_inference():
    st.header("Resume Inference")
    st.info("This section simplifies the inference process. "
            "Choose a test resume and observe the label that your trained pipeline will predict.")

    sample = st.selectbox(
        "Resume samples for inference",
        tuple(LABELS_MAP.values()),
        index=None,
        placeholder="Select a resume sample",
    )
    infer = st.button('Run Inference')

    if infer:
        run_inference(sample)

def run_inference(selected_sample):
    with st.spinner('Running inference...'):
        try:
            sample_file = "_".join(selected_sample.upper().split()) + ".txt"
            with open(SAMPLES_PATH / sample_file, encoding="utf-8") as file:
                sample_text = file.read()

            result = requests.post(
                'http://localhost:9000/api/inference',
                json={'text': sample_text}
            )
            label = LABELS_MAP.get(int(float(result.text)))
            st.success('Inference done!')
            st.metric(label="Status", value=f"Resume label: {label}")
        except Exception as e:
            display_error_message('Failed to call Inference API!', e)

if __name__ == "__main__":
    main()

