import streamlit as st
import torch
from torch import load as torch_load
from building_blocks import Transformer as TransformerClass  # your transformer class
import tools

# Set page configuration

st.set_page_config(
    page_title="🚀 Transformer Translator",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Configuration constants
CHECKPOINT_PATH = "checkpoints/model_epoch_400.pth"  # update the path as needed
EMBED_DIM = 300
MAX_LENGTH = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model (and vocabulary) from checkpoint
test_model, src_vocab, trg_vocab = tools.load_test_model(CHECKPOINT_PATH, TransformerClass, EMBED_DIM, DEVICE, MAX_LENGTH)


# --- Streamlit User Interface ---

# Custom CSS for a modern look
st.markdown(
    """
    <style>
    /* Adjust padding and background */
    .reportview-container .main .block-container{
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    /* Customize the text input box */
    .stTextInput>div>div>input {
        font-size: 1.1rem;
        padding: 0.8rem;
        border: 2px solid #4CAF50;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True
)

# # Sidebar (optional)
# st.sidebar.header("🛠️ Options")
# st.sidebar.markdown("You can add more settings here!")
# st.sidebar.image("https://via.placeholder.com/300x150.png?text=Translator+App", use_column_width=True)

# Main Interface
st.title("🚀 Transformer Translator")
st.markdown("### Translate an English sentence into German with confidence! 😎💬")

text_input = st.text_input("Enter your sentence here:", placeholder="e.g., Good morning")

if st.button("Translate Now 🚀"):
    if text_input.strip() == "":
        st.warning("⚠️ Please enter a sentence!")
    else:
        with st.spinner("Translating... please wait ⏳"):
            translation = tools.translate(text_input, test_model, src_vocab, trg_vocab, DEVICE, MAX_LENGTH)
        st.success(f"**Translation:** {translation}")
        st.balloons()  # adds a fun celebratory effect
