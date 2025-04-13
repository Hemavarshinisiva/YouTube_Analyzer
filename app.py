# YouTube Comment Sentiment Analyzer (Gradio using YouTube Data API)

import gradio as gr
from transformers import pipeline
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import emoji

# Load sentiment analysis pipeline
sentiment_model = pipeline("sentiment-analysis")

# Set your YouTube Data API key directly here (replace with your actual key)
YOUTUBE_API_KEY = "AIzaSyDPlgo45iiwS7slK5MGvyvlfHa28w_ip1s"
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Extract video ID from URL
def extract_video_id(url):
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    elif query.hostname in ('www.youtube.com', 'youtube.com'):
        if query.path == '/watch':
            return parse_qs(query.query).get('v', [None])[0]
        elif query.path[:7] == '/embed/':
            return query.path.split('/')[2]
        elif query.path[:3] == '/v/':
            return query.path.split('/')[2]
    return None

# Fetch comments using YouTube Data API
def fetch_comments(video_id, limit):
    comments = []
    try:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(limit, 100),
            textFormat="plainText"
        ).execute()

        while response and len(comments) < limit:
            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)
                if len(comments) >= limit:
                    break
            if "nextPageToken" in response:
                response = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=min(limit - len(comments), 100),
                    pageToken=response["nextPageToken"],
                    textFormat="plainText"
                ).execute()
            else:
                break
    except Exception as e:
        return [], f"Error fetching comments: {str(e)}"
    return comments, None

# Convert emojis to text description
def preprocess_comment(comment):
    return emoji.demojize(comment, delimiters=(" ", " ")).replace("_", " ")

# Analyze sentiments of comments and return summary, details, and chart
def analyze_youtube_comments(url, limit):
    video_id = extract_video_id(url)
    comments, error = fetch_comments(video_id, limit)

    if error:
        return error, pd.DataFrame(columns=["Comment", "Sentiment", "Confidence"]), plt.figure()

    if not comments:
        return "No comments fetched.", pd.DataFrame(columns=["Comment", "Sentiment", "Confidence"]), plt.figure()

    results = []
    sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}

    for comment in comments:
        preprocessed = preprocess_comment(comment)
        result = sentiment_model(preprocessed)[0]
        label = result['label']
        score = result['score']
        if label not in sentiment_counts:
            sentiment_counts[label] = 0
        sentiment_counts[label] += 1
        results.append({"Comment": comment, "Sentiment": label, "Confidence": round(score, 2)})

    df = pd.DataFrame(results)
    total = len(comments)
    sentiment_summary = f"Positive: {sentiment_counts.get('POSITIVE', 0)}, Negative: {sentiment_counts.get('NEGATIVE', 0)}, Neutral: {sentiment_counts.get('NEUTRAL', 0)}, Total: {total}"

    # Plot sentiment distribution
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='Sentiment', palette='pastel', ax=ax)
    ax.set_title('Sentiment Distribution')
    ax.set_ylabel('Count')
    plt.tight_layout()

    return sentiment_summary, df, fig

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🎥 YouTube Comment Sentiment Analyzer")
    with gr.Row():
        url_input = gr.Textbox(label="YouTube Video URL")
        comment_limit = gr.Slider(minimum=5, maximum=100, value=20, step=1, label="Number of Comments")
        analyze_btn = gr.Button("Analyze")
        clear_btn = gr.Button("Clear")

    sentiment_stats = gr.Textbox(label="Sentiment Summary")
    comment_table = gr.Dataframe(headers=["Comment", "Sentiment", "Confidence"], label="Comment Analysis")
    sentiment_plot = gr.Plot(label="Sentiment Chart")

    analyze_btn.click(analyze_youtube_comments, inputs=[url_input, comment_limit], outputs=[sentiment_stats, comment_table, sentiment_plot])
    clear_btn.click(fn=lambda: ("", pd.DataFrame(columns=["Comment", "Sentiment", "Confidence"]), plt.figure()), inputs=[], outputs=[sentiment_stats, comment_table, sentiment_plot])

demo.launch()
