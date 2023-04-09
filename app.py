import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
import os
from paddleocr import PaddleOCR
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain import PromptTemplate
from ast import literal_eval

# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False
    st.session_state.placeholder = "Biotin, Digoxin, Aspirin"
    result_interact = "The list of drugs is not included!"


os.environ["OPENAI_API_KEY"] = "PUT_YOUR_API_KEY"

result = {'drug': '', 'dose': {'value': '', 'unit': ''}}

# Given a text is containing information about drug labels, extract name of 'drug' and value and unit of 'dose' mentioned in the text in JSON format .

prompt_extract = """
The below text is related to drug labels. extract name of 'drug' and value and unit of 'dose' in JSON format. Don't explain extra.

Text: {text}
"""

model_name = "gpt-3.5-turbo"
llm = OpenAI(temperature=0.0, model_name=model_name)

prompt_template_extract = PromptTemplate(
    input_variables=["text"], template=prompt_extract
)

prompt_interact = """
Compare the list of medications provided: {list_of_drugs}. 
Using your knowledge of drug interactions, 
identify any potential interactions that could occur if these medications were taken together. 
Write in summary. Don't explain more.
"""

prompt_template_interact = PromptTemplate(
    input_variables=["list_of_drugs"], template=prompt_interact
)

ocr_model = PaddleOCR(lang='en', use_gpu=False, use_angle_cls=True, det_db_box_thresh=0.5, det_db_thresh=0.5, 
                        show_log=False, use_onnx=False)


st.markdown(
    """
    <style>
        button[title^=Exit]+div [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #0099ff;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #00ff00;
    color:#ff0000;
    }
</style>""", unsafe_allow_html=True)

width = st.slider('What is the width in pixels?', 0, 600, 400)

def load_image():
    uploaded_file = st.file_uploader(label='Pick an drug box image')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data, caption='', width=width)
        return image_data
    
    
st.title('MedicifyMe Demo')
image = load_image()
if  image is not None:
    image = np.asarray(Image.open(BytesIO(image)).convert('RGB'))
if st.button('Process Image'):
    result = ocr_model.ocr(image)
    texts = [res[1][0] for res in result[0] if len(res[1][0]) > 1]
    result = llm(prompt_template_extract.format(text=",".join(texts)))
    print("result: ", result)
    result = literal_eval(result)
    result['drug'] = " ".join(result['drug'].split(" ")[:2])

col1, col2, col3 = st.columns(3)
with col1:
    st.text_input(
        "Drug Name",
        result['drug'],
        key="placeholder1",
        disabled=True
    )
with col2:
    st.text_input(
        "Dose Value",
        result['dose']['value'],
        key="placeholder2",
        disabled=True
    )

with col3:
    st.text_input(
        "Dose Unit",
        result['dose']['unit'],
        key="placeholder3",
        disabled=True
    )

text_input = st.text_input(
        "Enter list of drugs 👇",
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
        placeholder=st.session_state.placeholder,
    )

if st.button('Checker') and text_input:
    drugs_name = text_input.split(",")
    result_interact = llm(prompt_template_interact.format(list_of_drugs=drugs_name))
    st.success(result_interact)

