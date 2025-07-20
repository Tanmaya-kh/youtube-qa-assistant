import streamlit as st
import openai
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

# Load OpenAI API key from secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("📺 YouTube Video QA Assistant")
st.write("Ask questions about the content of a YouTube video.")

# Helper to extract video ID
def extract_video_id(url_or_id):
    if "youtube.com" in url_or_id or "youtu.be" in url_or_id:
        try:
            parsed_url = urlparse(url_or_id)
            if parsed_url.hostname == "youtu.be":
                return parsed_url.path[1:]
            elif parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
                query = parse_qs(parsed_url.query)
                return query.get("v", [None])[0]
        except Exception:
            return None
    return url_or_id  # assume it's a raw video ID

# Get transcript from video
@st.cache_data(show_spinner=False)
def get_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join([t["text"] for t in transcript])

# QA function
def ask_question(transcript, question):
    prompt = (
        "You are a helpful assistant. Given the following transcript of a YouTube video, "
        "answer the user's question based on the video content.\n\n"
        f"Transcript:\n{transcript}\n\n"
        f"Question: {question}\nAnswer:"
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response["choices"][0]["message"]["content"]

# UI inputs
video_input = st.text_input("Enter YouTube Video URL or ID:")
question_input = st.text_input("Ask a question about the video:")

if video_input and question_input:
    try:
        with st.spinner("Fetching transcript..."):
            video_id = extract_video_id(video_input)
            transcript = get_transcript(video_id)

        with st.spinner("Thinking..."):
            answer = ask_question(transcript, question_input)

        st.success("Answer:")
        st.write(answer)

    except Exception as e:
        st.error(f"Error: {e}")
