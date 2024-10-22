import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
from PIL import Image

# Load PwC logo
def load_logo():
    return Image.open("/home/maximilian.laechelin/pwcProject/Logo-pwc.png")

# Load the pkl files from the specified directory
@st.cache_data
def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Ensure event dates are parsed as datetime
def parse_events(events):
    return {pd.to_datetime(date): label for date, label in events.items()}

# Plot neutral and negative comment counts using Plotly
def plot_neutral_negative_counts(df, events):
    df['comment_date'] = pd.to_datetime(df['comment_date'])
    
    neutral_counts = df[df['bertweet_sentiment_class'] == 'neutral'].set_index('comment_date').resample('M').size()
    negative_counts = df[df['bertweet_sentiment_class'] == 'negative'].set_index('comment_date').resample('M').size()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=neutral_counts.index, y=neutral_counts, mode='lines+markers', name='Neutral Comments', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=negative_counts.index, y=negative_counts, mode='lines+markers', name='Negative Comments', line=dict(color='red')))

    for i, (event_date, event_label) in enumerate(events.items()):
        fig.add_vline(x=event_date, line=dict(color='gray', dash='dash'))
        fig.add_annotation(
            x=event_date, 
            y=max(neutral_counts.max(), negative_counts.max()), 
            text=event_label, 
            showarrow=True, 
            arrowhead=2, 
            ax=0, 
            ay=-40 - (i % 2) * 20,  # Alternate annotation positioning to reduce overlap
            bordercolor="black",
            borderwidth=1
        )

    fig.update_layout(title='Monthly Neutral and Negative Comment Counts',
                      xaxis_title='Date',
                      yaxis_title='Count',
                      template='plotly_white')
    st.plotly_chart(fig)

# Plot average sentiment score using Plotly
def plot_average_sentiment(df, events):
    df['comment_date'] = pd.to_datetime(df['comment_date'])
    
    monthly_bertweet_sentiment = df.set_index('comment_date').resample('M')['bertweet_sentiment_score'].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_bertweet_sentiment['comment_date'], y=monthly_bertweet_sentiment['bertweet_sentiment_score'], mode='lines+markers', name='Average BERTweet Sentiment', line=dict(color='green')))

    for i, (event_date, event_label) in enumerate(events.items()):
        fig.add_vline(x=event_date, line=dict(color='gray', dash='dash'))
        fig.add_annotation(
            x=event_date, 
            y=monthly_bertweet_sentiment['bertweet_sentiment_score'].max(), 
            text=event_label, 
            showarrow=True, 
            arrowhead=2, 
            ax=0, 
            ay=-40 - (i % 2) * 20,  # Alternate annotation positioning to reduce overlap
            bordercolor="black",
            borderwidth=1
        )

    fig.update_layout(title='Monthly Average Sentiment Score (BERTweet)',
                      xaxis_title='Date',
                      yaxis_title='Average Sentiment Score',
                      template='plotly_white')
    st.plotly_chart(fig)

# Streamlit layout
st.set_page_config(page_title="YouTube Comment Sentiment Analysis", page_icon=load_logo(), layout="wide")

st.image(load_logo(), width=150)
st.title("YouTube Comment Sentiment Analysis")

# Sidebar for settings
st.sidebar.title("Settings")
dataset_option = st.sidebar.selectbox("Choose a dataset to visualize:", ("Audi", "Volkswagen", "Porsche", "BMW"))

# Load corresponding dataset based on selection
if dataset_option == "Audi":
    df = load_data('/home/maximilian.laechelin/pwcProject/data/audi_analyzed.pkl')
elif dataset_option == "Volkswagen":
    df = load_data('/home/maximilian.laechelin/pwcProject/data/volkswagen_analyzed.pkl')
elif dataset_option == "BMW":
    df = load_data('/home/maximilian.laechelin/pwcProject/data/bmw_analyzed.pkl')
elif dataset_option == "Porsche":
    df = load_data('/home/maximilian.laechelin/pwcProject/data/porsche_analyzed.pkl')

# Define industry events based on the selected dataset
if dataset_option == "Audi":
    events = {
        '2018-06-03': 'Audi CEO Arrested (Dieselgate)',
        '2019-09-10': 'Audi e-Tron Electric Car Launch',
        '2020-01-15': 'Audi and Huawei Partnership',
        '2021-09-30': 'Audi Announces EV Strategy',
        '2023-03-01': 'Audi Self-Driving Car Announcement',
        '2024-07-15': 'Audi Wins Major EV Award'
    }
elif dataset_option == "Volkswagen":
    events = {
        '2015-09-18': 'Volkswagen Dieselgate Scandal',
        '2018-09-20': 'Volkswagen Electric Car Launch',
        '2020-03-02': 'Volkswagen and Microsoft Partnership',
        '2021-08-30': 'Volkswagen Announces EV Strategy',
        '2023-03-25': 'Volkswagen Launches New ID.4 SUV'
    }
elif dataset_option == "BMW":
    events = {
        '2016-11-15': 'BMW Launches New 5 Series',
        '2018-03-07': 'BMW and Microsoft Partnership',
        '2020-06-10': 'BMW iNext EV Reveal',
        '2021-07-01': 'BMW Announces EV Strategy',
        '2023-05-15': 'BMW Launches New i4 Electric Car'
    }
elif dataset_option == "Porsche":
    events = {
        '2018-09-04': 'Porsche Taycan Electric Car Launch',
        '2019-09-12': 'Porsche Mission E Concept Reveal',
        '2020-02-18': 'Porsche Joins Formula E',
        '2021-08-25': 'Porsche 911 GT3 Launch',
        '2023-03-17': 'Porsche 718 Spyder RS Announcement'
    }

# Ensure events are parsed as datetime
events = parse_events(events)

# Display KPIs
st.subheader("Key Performance Indicators")
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Total Comments", value=len(df))
with col2:
    st.metric(label="Average Sentiment Score", value=f"{df['bertweet_sentiment_score'].mean():.2f}")

# Display the plots using tabs
tab1, tab2 = st.tabs(["Neutral/Negative Counts", "Average Sentiment Score"])
with tab1:
    st.subheader(f"Neutral and Negative Comment Counts for {dataset_option}")
    plot_neutral_negative_counts(df, events)

with tab2:
    st.subheader(f"Average Sentiment Score for {dataset_option}")
    plot_average_sentiment(df, events)
