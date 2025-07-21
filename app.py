import os
import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
import re
from urllib.parse import urlparse, parse_qs

# Load OpenAI API Key from secret
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Prompt Template
prompt = PromptTemplate(
    template="""
You are a helpful and knowledgeable assistant.
Use the following transcript to answer the user's question accurately and confidently.
- If the answer is available in the transcript, answer directly without saying “the transcript says.”
- If the question is hypothetical or goes beyond the transcript, use logical reasoning based on the concepts in the context. Do not mention that the transcript doesn't contain the answer.
- Always respond clearly and conversationally, using simple formatting (no LaTeX or special syntax).
- When making calculations or estimates, explain them using clean, readable language.
Transcript:
{context}
Question: {question}
""",
    input_variables=['context', 'question']
)

def get_video_id_from_url(url_or_id):
    # If user already entered just the ID, return it
    if len(url_or_id) == 11 and re.match(r'^[a-zA-Z0-9_-]{11}$', url_or_id):
        return url_or_id

    try:
        parsed_url = urlparse(url_or_id)
        if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
            query = parse_qs(parsed_url.query)
            return query["v"][0] if "v" in query else None
        elif parsed_url.hostname in ["youtu.be"]:
            return parsed_url.path.lstrip("/")
        else:
            return None
    except Exception:
        return None

def answer_from_video(video_url_or_id, question):
    try:
        # Extract video ID
        video_id = get_video_id_from_url(video_url_or_id)
        if not video_id:
            return "⚠️ Could not extract video ID from the provided URL or input."

        # Step 1: Get transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)

        # Step 2: Chunk it
        chunks = splitter.create_documents([transcript])

        # Step 3: Embed + store
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        # Step 4: Retrieve relevant content
        relevant_docs = retriever.invoke(question)
        context_text = "\n\n".join(doc.page_content for doc in relevant_docs)

        # Step 5: Ask GPT
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        final_prompt = prompt.invoke({"context": context_text, "question": question})
        answer = llm.invoke(final_prompt)

        return answer.content

    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🎬 YouTube Comment Reply Assistant")
    gr.Markdown("Enter a YouTube URL or ID and a learner's question to get a smart, grounded reply.")

    with gr.Row():
        video_input = gr.Textbox(label="YouTube URL or Video ID", placeholder="e.g. https://youtu.be/wCEkK1YzqBo")
        question = gr.Textbox(label="Learner's Question", placeholder="e.g. What if I had 5 channels?")

    submit = gr.Button("Generate Answer")
    
    response = gr.Textbox(
        label="Suggested Reply", 
        lines=6, 
        interactive=False, 
        show_copy_button=True  # 👈 adds the nice double-rectangle icon
    )

    submit.click(fn=answer_from_video, inputs=[video_input, question], outputs=response)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))

