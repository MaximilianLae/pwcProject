import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
from PIL import Image
# from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load PwC logo
def load_logo():
    return Image.open("Logo-pwc.png")

# Load the pkl files from the specified directory
@st.cache_data
def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Ensure event dates are parsed as datetime
def parse_events(events):
    return {pd.to_datetime(date): label for date, label in events.items()}

# Plot neutral, negative counts and total comment count per month using Plotly
def plot_neutral_negative_counts(df, events):
    df['comment_date'] = pd.to_datetime(df['comment_date'])
    
    # Resample both sentiment counts and total comment count using the same timeline
    neutral_counts = df[df['bertweet_sentiment_class'] == 'neutral'].set_index('comment_date').resample('M').size()
    negative_counts = df[df['bertweet_sentiment_class'] == 'negative'].set_index('comment_date').resample('M').size()
    total_comment_count = df.set_index('comment_date').resample('M').size()

    # Plot Neutral and Negative Counts
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=neutral_counts.index, y=neutral_counts, mode='lines+markers', name='Neutral Comments', line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=negative_counts.index, y=negative_counts, mode='lines+markers', name='Negative Comments', line=dict(color='red')))

    for i, (event_date, event_label) in enumerate(events.items()):
        fig1.add_vline(x=event_date, line=dict(color='gray', dash='dash'))
        fig1.add_annotation(
            x=event_date, 
            y=max(neutral_counts.max(), negative_counts.max()), 
            text=event_label, 
            showarrow=True, 
            arrowhead=2, 
            ax=0, 
            ay=-40 - (i % 2) * 20,
            bordercolor="black",
            borderwidth=1
        )

    fig1.update_layout(title='Monthly Neutral and Negative Comment Counts',
                      xaxis_title='Date',
                      yaxis_title='Count',
                      template='plotly_white')
    st.plotly_chart(fig1, key='neutral_negative_counts')

    # Plot Total Comment Count
    plot_total_comment_count(total_comment_count, events, key='total_comment_count')

# Plot total comment count per month with events using Plotly
def plot_total_comment_count(total_comment_count, events, key):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=total_comment_count.index, y=total_comment_count, name='Total Comment Count', marker=dict(color='orange')))
    
    for event_date, event_label in events.items():
        fig.add_vline(x=event_date, line=dict(color='gray', dash='dash'))
        fig.add_annotation(
            x=event_date,
            y=total_comment_count.max(),
            text=event_label,
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            bordercolor="black",
            borderwidth=1
        )
    
    fig.update_layout(title='Total Comment Count Per Month',
                       xaxis_title='Month-Year',
                       yaxis_title='Total Comment Count',
                       template='plotly_white')

    # Display the total comment count plot in Streamlit
    st.plotly_chart(fig, key=key)

# Plot average sentiment score using Plotly
def plot_average_sentiment(df, events):
    df['comment_date'] = pd.to_datetime(df['comment_date'])
    
    monthly_bertweet_sentiment = df.set_index('comment_date').resample('M')['bertweet_sentiment_score'].mean().reset_index()
    total_comment_count = df.set_index('comment_date').resample('M').size()

    # Plot Average Sentiment Score
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
            ay=-40 - (i % 2) * 20,
            bordercolor="black",
            borderwidth=1
        )

    fig.update_layout(title='Monthly Average Sentiment Score (BERTweet)',
                      xaxis_title='Date',
                      yaxis_title='Average Sentiment Score',
                      template='plotly_white')
    st.plotly_chart(fig, key='average_sentiment')

    # Plot Total Comment Count (reuse the existing function)
    plot_total_comment_count(total_comment_count, events, key='total_comment_count_average_sentiment')

# Plot trust trends over time, average trust score, and comment count using Plotly
def plot_trust_analysis(df, events):
    df['comment_date'] = pd.to_datetime(df['comment_date'])

    # Trust Trends Over Time
    trust_trend = df.groupby(['comment_date', 'trust_classification']).size().unstack(fill_value=0).resample('M').sum()

    # Plot Trust Trends
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=trust_trend.index, y=trust_trend.get('This comment expresses trust in the brand', [0]*len(trust_trend)), mode='lines+markers', name='Trust', line=dict(color='green')))
    fig1.add_trace(go.Scatter(x=trust_trend.index, y=trust_trend.get('This comment is neutral', [0]*len(trust_trend)), mode='lines+markers', name='Neutral', line=dict(color='orange')))
    fig1.add_trace(go.Scatter(x=trust_trend.index, y=trust_trend.get('This comment expresses distrust towards the brand', [0]*len(trust_trend)), mode='lines+markers', name='Distrust', line=dict(color='red')))
    
    for event_date, event_label in events.items():
        fig1.add_vline(x=event_date, line=dict(color='gray', dash='dash'))
        fig1.add_annotation(
            x=event_date,
            y=max(trust_trend.max()),
            text=event_label,
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            bordercolor="black",
            borderwidth=1
        )
    
    fig1.update_layout(title='Trust Trends Over Time',
                       xaxis_title='Month-Year',
                       yaxis_title='Count of Comments',
                       template='plotly_white')
    
    # Average Trust Score Over Time
    avg_trust_score = df.set_index('comment_date').resample('M')['trust_score'].mean()

    # Plot Average Trust Score
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=avg_trust_score.index, y=avg_trust_score, mode='lines+markers', name='Average Trust Score', line=dict(color='blue')))
    
    for event_date, event_label in events.items():
        fig2.add_vline(x=event_date, line=dict(color='gray', dash='dash'))
        fig2.add_annotation(
            x=event_date,
            y=avg_trust_score.max(),
            text=event_label,
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            bordercolor="black",
            borderwidth=1
        )
    
    fig2.update_layout(title='Average Trust Score Over Time',
                       xaxis_title='Month-Year',
                       yaxis_title='Average Trust Score',
                       template='plotly_white',
                       yaxis_range=[-1, 1])

    # Total Comment Count Per Month
    comment_count = df.set_index('comment_date').resample('M').size()

    # Plot Total Comment Count
    plot_total_comment_count(comment_count, events, key='total_comment_count_trust')

    # Display the figures in Streamlit
    st.plotly_chart(fig1, key='trust_trends')
    st.plotly_chart(fig2, key='average_trust_score')

    # Display 10 examples of trust and 10 examples of distrust
    st.subheader("Examples of Trust and Distrust Comments")

    # Extract 10 random examples of comments expressing trust
    trust_examples = df[df['trust_classification'] == "This comment expresses trust in the brand"].sample(n=10, random_state=1)
    if len(trust_examples) > 0:
        st.markdown("### Comments Expressing Trust")
        for _, row in trust_examples.iterrows():
            st.markdown(f"- {row['comment_text']}")

    # Extract 10 random examples of comments expressing distrust from rows 107-117
    distrust_examples = df[df['trust_classification'] == "This comment expresses distrust towards the brand"].iloc[107:117]
    if len(distrust_examples) > 0:
        st.markdown("### Comments Expressing Distrust")
        for _, row in distrust_examples.iterrows():
            st.markdown(f"- {row['comment_text']}")

# Streamlit layout
st.set_page_config(page_title="YouTube Comment Sentiment Analysis", page_icon=load_logo(), layout="wide")

st.image(load_logo(), width=150)
st.title("YouTube Comment Sentiment Analysis")

# Sidebar for settings
st.sidebar.title("Settings")
dataset_option = st.sidebar.selectbox("Choose a dataset to visualize:", ("Audi", "Volkswagen", "Porsche", "BMW"))

# Load corresponding dataset based on selection
if dataset_option == "Audi":
    df = load_data('data/audi_analyzed.pkl')
    df_classified = load_data('data/audi_analyzed_classified.pkl')
elif dataset_option == "Volkswagen":
    df = load_data('data/volkswagen_analyzed.pkl')
    df_classified = load_data('data/volkswagen_analyzed_classified.pkl')
elif dataset_option == "BMW":
    df = load_data('data/bmw_analyzed.pkl')
    df_classified = None  # No classified data available
elif dataset_option == "Porsche":
    df = load_data('data/porsche_analyzed.pkl')
    df_classified = None  # No classified data available

# Add trust score column for classified datasets
if df_classified is not None:
    df_classified['trust_score'] = df_classified['trust_classification'].map({
        "This comment expresses trust in the brand": 1,
        "This comment expresses distrust towards the brand": -1,
        "This comment is neutral": 0
    })

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

# Calculate Correlations
total_comment_count = df.set_index('comment_date').resample('M').size()
average_sentiment_score = df.set_index('comment_date').resample('M')['bertweet_sentiment_score'].mean()
sentiment_correlation = total_comment_count.corr(average_sentiment_score)

if df_classified is not None and 'trust_score' in df_classified.columns:
    average_trust_score = df_classified.set_index('comment_date').resample('M')['trust_score'].mean()
    trust_correlation = total_comment_count.corr(average_trust_score)
else:
    trust_correlation = None

# Display KPIs
st.subheader("Key Performance Indicators")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Comments", value=len(df))
with col2:
    st.metric(label="Average Sentiment Score", value=f"{df['bertweet_sentiment_score'].mean():.2f}")
with col3:
    st.metric(label="Sentiment-Comment Correlation", value=f"{sentiment_correlation:.2f}")

if trust_correlation is not None:
    st.metric(label="Trust-Comment Correlation", value=f"{trust_correlation:.2f}")

# Display the plots using tabs
tab1, tab2, tab3 = st.tabs(["Neutral/Negative Counts", "Average Sentiment Score", "Trust Analysis"])
with tab1:
    st.subheader(f"Neutral and Negative Comment Counts for {dataset_option}")
    plot_neutral_negative_counts(df, events)

with tab2:
    st.subheader(f"Average Sentiment Score for {dataset_option}")
    plot_average_sentiment(df, events)

with tab3:
    if df_classified is not None:
        st.subheader(f"Trust Analysis for {dataset_option}")
        plot_trust_analysis(df_classified, events)
    else:
        st.warning("Trust analysis is only available for Audi and Volkswagen.")
