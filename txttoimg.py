import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Set your API keys from environment variables
HF_API_KEY = os.getenv("HF_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Load the Stable Diffusion pipeline
@st.cache(allow_output_mutation=True)
def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, use_auth_token=HF_API_KEY)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe.to("cuda")

pipe = load_pipeline()

# Initialize the ChatGoogleGenerativeAI model with the API key
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

def get_enhanced_prompt(user_prompt):
    # Generate an enhanced prompt using LangChain and Gemini API
    response = llm.invoke(user_prompt)
    return format_response(response)

def format_response(response):
    # Extract content from the AIMessage object and format it
    content = response.content
    return content.replace('\\n', '\n')

st.title("Stable Diffusion Image Generator with Enhanced Prompts")
st.write("Enter a prompt to generate an image using Stable Diffusion and Google Generative AI for prompt enhancement.")

user_prompt = st.text_input("Enter your prompt here", "")

if st.button("Generate Image"):
    if user_prompt:
        with st.spinner("Enhancing prompt..."):
            enhanced_prompt = get_enhanced_prompt(user_prompt)

            with st.spinner("Generating image..."):
                image = pipe(enhanced_prompt).images[0]
                st.image(image, caption="Generated Image")
                image.save("generated_image.png")
                st.success("Image generated and saved as generated_image.png")
    else:
        st.error("Please enter a prompt to generate an image.")
