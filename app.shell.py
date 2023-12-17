from tiktoken import encoding_for_model
from weaviate_interface import WeaviateClient
from prompt_templates import question_answering_prompt_series, question_answering_system
from openai_interface import GPT_Turbo
from app_features import convert_seconds, generate_prompt_series, validate_token_threshold, load_data
from reranker import ReRanker
from loguru import logger 
import streamlit as st
import css_templates
import sys
import json
import os

# load environment variables
from dotenv import load_dotenv
load_dotenv('.env', override=True)
 
## PAGE CONFIGURATION
st.set_page_config(page_title="Impact Theory", 
                   page_icon=None, 
                   layout="wide", 
                   initial_sidebar_state="auto", 
                   menu_items=None)

data_path = './data/impact_theory_data.json'

impact_theory_data = load_data(data_path)

## GUESTS
guest_list = list(set([data['guest'] for data in impact_theory_data]))

## RETRIEVER
api_key = os.environ['WEAVIATE_API_KEY']
url = os.environ['WEAVIATE_ENDPOINT']

client = WeaviateClient(api_key, url)

## RERANKER
reranker = ReRanker()

## LLM
model = 'gpt-4-1106-preview'
llm = GPT_Turbo(model=model)

## ENCODING
encoding = encoding_for_model(model)

## INDEX NAME
class_name = 'Impact_theory_minilm_256'

def main():
    st.write(css_templates.load_css(), unsafe_allow_html=True)
    
    with st.sidebar:
        guest = st.selectbox('Select Guest', options=guest_list, index=None, placeholder='Select Guest')

    st.image('./assets/impact-theory-logo.png', width=400)
    st.subheader(f"Chat with the Impact Theory podcast: ")
    st.write('\n')
    col1, _ = st.columns([7,3])
    with col1:
        query = st.text_input('Enter your question: ')
        st.write('\n\n\n\n\n')

        if query:
            where_filter = None

            if guest:
                # See: https://weaviate.io/developers/weaviate/api/graphql/filters#single-operand-condition
                where_filter = {
                    "path": ["guest"],
                    "operator": "Equal",
                    "valueText": guest
                }

            display_properties = ['title', 'video_id', 'length', 'thumbnail_url', 'views', 'episode_url',
                                    'doc_id', 'guest', 'content', 'summary']

            # make hybrid call to weaviate
            hybrid_response = client.hybrid_search(query, class_name=class_name, properties=['content', 'summary', 'title', 'guest'],
                                                   alpha=0.3,
                                                    display_properties=display_properties,
                                                    where_filter=where_filter,
                                                    limit=200)
            # rerank results
            ranked_response = reranker.rerank(hybrid_response, query)

            # validate token count is below threshold
            valid_response = validate_token_threshold(ranked_response, 
                                                       question_answering_prompt_series, 
                                                       query=query,
                                                       tokenizer=encoding, # variable from ENCODING,
                                                       token_threshold=4000, 
                                                       verbose=True)

            # # generate LLM prompt
            prompt = generate_prompt_series(query=query, results=valid_response)
            
            # prep for streaming response
            st.subheader("Response from Impact Theory (context)")
            with st.spinner('Generating Response...'):
                st.markdown("----")
                res_box = st.empty()
                report = []
                col_1, _ = st.columns([4, 3], gap='large')
                
                # execute chat call to LLM                
                stream_resp = llm.get_chat_completion(
                    prompt,
                    system_message=question_answering_system,
                    temperature=0.9,
                    stream=True,
                    show_response=True
                )

                try:
                    with res_box:
                        # Stream the response from OpenAI
                        # See: https://pypi.org/project/openai/
                        for chunk in stream_resp:
                            resp = chunk.choices[0].delta.content
                            if resp:
                                report.append(resp)
                                result = "".join(report).strip()
                                res_box.markdown(f'*{result}*')

                except Exception as e:
                    print(e)

            st.subheader("Search Results")
            for i, hit in enumerate(valid_response):
                col1, col2 = st.columns([7, 3], gap='large')
                episode_num= i + 1 # get episode number
                image = hit['thumbnail_url'] # get thumbnail_url
                episode_url = hit['episode_url'] # get episode_url
                title = hit['title'] # get title
                show_length = hit['length'] # get length
                content = hit['content'] # get content
                time_string = convert_seconds(show_length)
                
                with col1:
                    st.write(css_templates.search_result(
                        i=i, 
                        url=episode_url,
                        episode_num=episode_num,
                        title=title,
                        content=content, 
                        length=time_string),
                        unsafe_allow_html=True
                    )
                    st.write('\n\n')
                with col2:
                    st.image(image, caption=title.split('|')[0], width=200, use_column_width=False)

if __name__ == '__main__':
    main()
