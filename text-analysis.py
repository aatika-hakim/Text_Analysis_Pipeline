# import
import os
from typing import TypedDict, List
import streamlit as st
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')

# Initialize Groq model with a LLaMA model name
groq_instance = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.5)

# state structure
class AnalysisState(TypedDict):
    input_text: str
    category: str
    entities: List[str]
    summary: str
    keywords: List[str]
    context: str

# analysis functions
def classify_text(state: AnalysisState):
    classification_prompt = PromptTemplate(
        input_variables=["input_text"],
        template="Classify the following text. Categories: News, Blog, Research, etc.\n\nText: {input_text}\n\nCategory:"
    )
    message = HumanMessage(content=classification_prompt.format(input_text=state["input_text"]))
    category_result = groq_instance.predict_messages([message]).content.strip()
    return {"category": category_result}

def extract_entities(state: AnalysisState):
    entity_prompt = PromptTemplate(
        input_variables=["input_text"],
        template="Identify and list all relevant entities from the text, including but not limited to Person, Organization, Location, Date, Event, and any other important concepts.\n\nText: {input_text}\n\nEntities:"
    )
    message = HumanMessage(content=entity_prompt.format(input_text=state["input_text"]))
    entities_result = groq_instance.predict_messages([message]).content.strip().split(", ")
    return {"entities": entities_result}

def generate_summary(state: AnalysisState):
    summary_prompt = PromptTemplate(
        input_variables=["input_text"],
        template="Provide a concise summary of this text.\n\nText: {input_text}\n\nSummary:"
    )
    message = HumanMessage(content=summary_prompt.format(input_text=state["input_text"]))
    summary_result = groq_instance.predict_messages([message]).content.strip()
    return {"summary": summary_result}

def identify_keywords(state: AnalysisState):
    keyword_prompt = PromptTemplate(
        input_variables=["input_text"],
        template="Extract important keywords from the following text.\n\nText: {input_text}\n\nKeywords:"
    )
    message = HumanMessage(content=keyword_prompt.format(input_text=state["input_text"]))
    keywords_result = groq_instance.predict_messages([message]).content.strip().split(", ")
    return {"keywords": keywords_result}

def determine_context(state: AnalysisState):
    context_prompt = PromptTemplate(
        input_variables=["input_text"],
        template="Identify the main context of this text. Possible themes include Business, Technology, etc.\n\nText: {input_text}\n\nContext:"
    )
    message = HumanMessage(content=context_prompt.format(input_text=state["input_text"]))
    context_result = groq_instance.predict_messages([message]).content.strip()
    return {"context": context_result}

# Set up the workflow
text_workflow = StateGraph(AnalysisState)

# Add distinct nodes for the analysis
text_workflow.add_node("classify_text", classify_text)
text_workflow.add_node("extract_entities", extract_entities)
text_workflow.add_node("generate_summary", generate_summary)
text_workflow.add_node("identify_keywords", identify_keywords)
text_workflow.add_node("determine_context", determine_context)

# workflow sequence
text_workflow.set_entry_point("classify_text")
text_workflow.add_edge("classify_text", "extract_entities")
text_workflow.add_edge("extract_entities", "generate_summary")
text_workflow.add_edge("generate_summary", "identify_keywords")
text_workflow.add_edge("identify_keywords", "determine_context")
text_workflow.add_edge("determine_context", END)

# Compile the graph
compiled_analysis = text_workflow.compile()

# Streamlit user interface
st.set_page_config(page_title="Text Analysis Pipeline", page_icon="📝", layout="wide")

st.markdown("<h1 style='text-align: center;'>📝 Text Analysis Pipeline</h1>", unsafe_allow_html=True)


st.markdown("<div style='text-align: center; margin-bottom: 10px;'>This tool analyzes text for detailed insights, including categorization, entity extraction, summarization, and keyword identification.</div>", unsafe_allow_html=True)


#user input
user_text = st.text_area("", placeholder="Enter your text for detailed analysis...", height=150, max_chars=2000)


if st.button("Start Analysis") and user_text:
    initial_data = {"input_text": user_text}
    analysis_results = compiled_analysis.invoke(initial_data)

    st.subheader("Analysis Results")

    # Display results 
    st.markdown(
        f"""
        <div style=" margin: 0 auto; width: 80%; background-color: #f0f8ff; padding: 10px; border-radius: 5px;">
            <strong>Category:</strong> {analysis_results['category']}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
        <div style=" margin: 0 auto; width: 80%; background-color: #e6ffe6; padding: 10px; border-radius: 5px;">
            <strong>Entities:</strong> {', '.join(analysis_results['entities'])}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
        <div style=" margin: 0 auto; width: 80%; background-color: #fff3e6; padding: 10px; border-radius: 5px;">
            <strong>Summary:</strong> {analysis_results['summary']}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
        <div style=" margin: 0 auto; width: 80%; background-color: #e6f7ff; padding: 10px; border-radius: 5px;">
            <strong>Keywords:</strong> {', '.join(analysis_results['keywords'])}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
        <div style=" margin: 0 auto; width: 80%; background-color: #fffbe6; padding: 10px; border-radius: 5px;">
            <strong>Context:</strong> {analysis_results['context']}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.success("Analysis completed successfully!")
