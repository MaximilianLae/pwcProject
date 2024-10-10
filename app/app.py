import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Load the pkl files from the specified directory
@st.cache_data
def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Ensure event dates are parsed as datetime
def parse_events(events):
    return {pd.to_datetime(date): label for date, label in events.items()}

# Updated plot_neutral_negative_counts function
def plot_neutral_negative_counts(df, events):
    df['comment_date'] = pd.to_datetime(df['comment_date'])
    
    # Count neutral and negative comments by month
    neutral_counts = df[df['bertweet_sentiment_class'] == 'neutral'].set_index('comment_date').resample('M').size()
    negative_counts = df[df['bertweet_sentiment_class'] == 'negative'].set_index('comment_date').resample('M').size()

    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(neutral_counts.index, neutral_counts, label='Neutral Comments', marker='o', color='blue')
    ax.plot(negative_counts.index, negative_counts, label='Negative Comments', marker='x', color='red')

    # Add vertical lines for events
    for event_date, event_label in events.items():
        ax.axvline(x=event_date, color='gray', linestyle='--', lw=2)
        ax.text(event_date + pd.Timedelta(days=10), plt.ylim()[1] * 0.9, event_label, rotation=90, verticalalignment='top', horizontalalignment='left')

    ax.set_xlabel('Date')
    ax.set_ylabel('Count')
    ax.set_title('Monthly Neutral and Negative Comment Counts')
    ax.legend()
    plt.tight_layout()

    # Pass the figure object to st.pyplot
    st.pyplot(fig)

# Updated plot_average_sentiment function
def plot_average_sentiment(df, events):
    df['comment_date'] = pd.to_datetime(df['comment_date'])
    
    # Aggregate average sentiment score per month
    monthly_bertweet_sentiment = df.set_index('comment_date').resample('M')['bertweet_sentiment_score'].mean().reset_index()

    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(monthly_bertweet_sentiment['comment_date'], monthly_bertweet_sentiment['bertweet_sentiment_score'], label='Average BERTweet Sentiment', marker='o', color='green')

    for event_date, event_label in events.items():
        ax.axvline(x=event_date, color='gray', linestyle='--', lw=2)
        ax.text(event_date + pd.Timedelta(days=10), plt.ylim()[1] * 0.9, event_label, rotation=90, verticalalignment='top', horizontalalignment='left')

    ax.set_xlabel('Date')
    ax.set_ylabel('Average Sentiment Score')
    ax.set_title('Monthly Average Sentiment Score (BERTweet)')
    plt.tight_layout()

    # Pass the figure object to st.pyplot
    st.pyplot(fig)

# Streamlit layout
st.title("YouTube Comment Sentiment Analysis")

# Dropdown to select the dataset
dataset_option = st.selectbox(
    "Choose a dataset to visualize:",
    ("Audi", "Volkswagen", "Porsche", "BMW")
)

""" # Load corresponding dataset based on selection
if dataset_option == "Audi":
    df = load_data('/home/maximilian.laechelin/pwcProject/audi_analyzed.pkl')
elif dataset_option == "Volkswagen":
    df = load_data('/home/maximilian.laechelin/pwcProject/volkswagen_analyzed.pkl')
elif dataset_option == "BMW":
    df = load_data('/home/maximilian.laechelin/pwcProject/bmw_analyzed.pkl')
elif dataset_option == "Porsche":
    df = load_data('/home/maximilian.laechelin/pwcProject/porsche_analyzed.pkl') """

# Load corresponding dataset based on selection
if dataset_option == "Audi":
    df = load_data('data/audi_analyzed.pkl')
elif dataset_option == "Volkswagen":
    df = load_data('data/volkswagen_analyzed.pkl')
elif dataset_option == "BMW":
    df = load_data('data/bmw_analyzed.pkl')
elif dataset_option == "Porsche":
    df = load_data('data/porsche_analyzed.pkl')


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

# Display the plots
st.subheader(f"Neutral and Negative Comment Counts for {dataset_option}")
plot_neutral_negative_counts(df, events)

st.subheader(f"Average Sentiment Score for {dataset_option}")
plot_average_sentiment(df, events)
