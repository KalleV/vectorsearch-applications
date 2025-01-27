#external files
from openai_interface import GPT_Turbo
from weaviate_interface import WeaviateClient
from llama_index.finetuning import EmbeddingQAFinetuneDataset
from prompt_templates import qa_generation_prompt
from reranker import ReRanker

#standard library imports
import json
import time
import uuid
import os
import re
import random
from datetime import datetime
from typing import List, Dict, Tuple, Union, Literal

#misc
from tqdm import tqdm


class QueryContextGenerator:
    '''
    Class designed for the generation of query/context pairs using a
    Generative LLM. The LLM is used to generate questions from a given
    corpus of text. The query/context pairs can be used to fine-tune 
    an embedding model using a MultipleNegativesRankingLoss loss function
    or can be used to create evaluation datasets for retrieval models.
    '''
    def __init__(self, openai_key: str, model_id: str='gpt-3.5-turbo-0613'):
        self.llm = GPT_Turbo(model=model_id, api_key=openai_key)

    def clean_validate_data(self,
                            data: List[dict], 
                            valid_fields: List[str]=['content', 'summary', 'guest', 'doc_id'],
                            total_chars: int=950
                            ) -> List[dict]:
        '''
        Strip original data chunks so they only contain valid_fields.
        Remove any chunks less than total_chars in size. Prevents LLM
        from asking questions from sparse content. 
        '''
        clean_docs = [{k:v for k,v in d.items() if k in valid_fields} for d in data]
        valid_docs = [d for d in clean_docs if len(d['content']) > total_chars]
        return valid_docs

    def train_val_split(self,
                        data: List[dict],
                        n_train_questions: int, 
                        n_val_questions: int, 
                        n_questions_per_chunk: int=2,
                        total_chars: int=950):
        '''
        Splits corpus into training and validation sets.  Training and 
        validation samples are randomly selected from the corpus. total_chars
        parameter is set based on pre-analysis of average doc length in the 
        training corpus. 
        '''
        clean_data = self.clean_validate_data(data, total_chars=total_chars)
        random.shuffle(clean_data)
        train_index = n_train_questions//n_questions_per_chunk
        valid_index = n_val_questions//n_questions_per_chunk
        end_index = valid_index + train_index
        if end_index > len(clean_data):
            raise ValueError('Cannot create dataset with desired number of questions, try using a larger dataset')
        train_data = clean_data[:train_index]
        valid_data = clean_data[train_index:end_index]
        print(f'Length Training Data: {len(train_data)}')
        print(f'Length Validation Data: {len(valid_data)}')
        return train_data, valid_data

    def generate_qa_embedding_pairs(
                                    self,
                                    data: List[dict],
                                    generate_prompt_tmpl: str=None,
                                    num_questions_per_chunk: int = 2,
                                    ) -> EmbeddingQAFinetuneDataset:
        """
        Generate query/context pairs from a list of documents. The query/context pairs
        can be used for fine-tuning an embedding model using a MultipleNegativesRankingLoss
        or can be used to create an evaluation dataset for retrieval models.
        """
        generate_prompt_tmpl = qa_generation_prompt if not generate_prompt_tmpl else generate_prompt_tmpl
        queries = {}
        relevant_docs = {}
        corpus = {chunk['doc_id'] : chunk['content'] for chunk in data}
        for chunk in tqdm(data):
            summary = chunk['summary']
            guest = chunk['guest']
            transcript = chunk['content']
            node_id = chunk['doc_id']
            query = generate_prompt_tmpl.format(summary=summary, 
                                                guest=guest,
                                                transcript=transcript,
                                                num_questions_per_chunk=num_questions_per_chunk)
            try:
                response = self.llm.get_chat_completion(prompt=query, temperature=0.1, max_tokens=100)
            except Exception as e:
                print(e)
                continue
            result = str(response).strip().split("\n")
            questions = [
                re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
            ]
            questions = [question for question in questions if len(question) > 0]

            for question in questions:
                question_id = str(uuid.uuid4())
                queries[question_id] = question
                relevant_docs[question_id] = [node_id]

        # construct dataset
        return EmbeddingQAFinetuneDataset(
            queries=queries, corpus=corpus, relevant_docs=relevant_docs
        )

def retrieval_evaluation(dataset: EmbeddingQAFinetuneDataset, 
                         class_name: str, 
                         retriever: WeaviateClient,
                         reranker: ReRanker=None,
                         alpha: float=0.5,
                         retrieve_limit: int=5,
                         results_top_k: int=5,
                         rerank_top_k: int=5,
                         chunk_size: int=256,
                         hnsw_config_keys: List[str]=['maxConnections', 'efConstruction', 'ef'],
                         search_type: Literal['kw', 'vector', 'hybrid', 'all']='all',
                         display_properties: List[str]=['doc_id', 'content'],
                         user_def_params: dict=None
                         ) -> Tuple[int, int, int]:
    '''
    Given a dataset, a retriever, and a reranker, evaluate the performance of the retriever and reranker. 
    Returns a dict of kw, vector, and hybrid hit rates and mrr scores. If inlude_miss_info is True, will
    also return a list of kw and vector responses and their associated queries that did not return a hit.

    Args:
    -----
    dataset: EmbeddingQAFinetuneDataset
        Dataset to be used for evaluation
    class_name: str
        Name of Class on Weaviate host to be used for retrieval
    retriever: WeaviateClient
        WeaviateClient object to be used for retrieval 
    reranker: ReRanker
        ReRanker model to be used for results reranking
    alpha: float=0.5
        Weighting factor for BM25 and Vector search.
        alpha can be any number from 0 to 1, defaulting to 0.5:
            alpha = 0 executes a pure keyword search method (BM25)
            alpha = 0.5 weighs the BM25 and vector methods evenly
            alpha = 1 executes a pure vector search method
    retrieve_limit: int=5
        Number of documents to retrieve from Weaviate host
    results_top_k: int=5
        Number of top results to evaluate
    rerank_top_k: int=5
        Number of top results to rerank
    chunk_size: int=256
        Number of tokens used to chunk text
    display_properties: List[str]=['doc_id', 'content']
        List of properties to be returned from Weaviate host for display in response
    '''
    if results_top_k > retrieve_limit:  # we don't want to retrieve less results than the top_k that we want to see returned
        retrieve_limit = results_top_k
        
    reranker_name = reranker.model_name if reranker else "None"
    
    results_dict = {'n':retrieve_limit, 
                    'top_k': results_top_k,
                    'alpha': alpha,
                    'Retriever': retriever.model_name_or_path, 
                    'Ranker': reranker_name,
                    'chunk_size': chunk_size,
                    'kw_hit_rate': 0,
                    'kw_mrr': 0,
                    'vector_hit_rate': 0,
                    'vector_mrr': 0,
                    'hybrid_hit_rate':0,
                    'hybrid_mrr': 0,
                    'total_misses': 0,
                    'total_questions':0
                    }
    #add extra params to results_dict
    results_dict = add_params(retriever, class_name, results_dict, user_def_params, hnsw_config_keys)
    if reranker:
        results_dict['rerank_top_k'] = rerank_top_k  # have to build the results_dict before we can add this information
        
    start = time.perf_counter()
    for query_id, q in tqdm(dataset.queries.items(), 'Queries'):
        results_dict['total_questions'] += 1
        
        #make Keyword, Vector, and Hybrid calls to Weaviate host
        try:
            kw_response = retriever.keyword_search(request=q, class_name=class_name, limit=retrieve_limit, display_properties=display_properties)
            vector_response = retriever.vector_search(request=q, class_name=class_name, limit=retrieve_limit, display_properties=display_properties)
            hybrid_response = retriever.hybrid_search(request=q, class_name=class_name, alpha=alpha, limit=retrieve_limit, display_properties=display_properties)           
            #rerank returned responses if reranker is provided
            if reranker:
                kw_response = reranker.rerank(kw_response, q, top_k=rerank_top_k)
                vector_response = reranker.rerank(vector_response, q, top_k=rerank_top_k)
                hybrid_response = reranker.rerank(hybrid_response, q, top_k=rerank_top_k)
            
            #collect doc_ids to check for document matches (include only results_top_k)
            kw_doc_ids = {result['doc_id']:i for i, result in enumerate(kw_response[:results_top_k], 1)}
            vector_doc_ids = {result['doc_id']:i for i, result in enumerate(vector_response[:results_top_k], 1)}
            hybrid_doc_ids = {result['doc_id']:i for i, result in enumerate(hybrid_response[:results_top_k], 1)}
            
            #extract doc_id for scoring purposes
            doc_id = dataset.relevant_docs[query_id][0]
     
            #increment hit_rate counters and mrr scores
            if doc_id in kw_doc_ids:
                results_dict['kw_hit_rate'] += 1
                results_dict['kw_mrr'] += 1/kw_doc_ids[doc_id]
            if doc_id in vector_doc_ids:
                results_dict['vector_hit_rate'] += 1
                results_dict['vector_mrr'] += 1/vector_doc_ids[doc_id]
            if doc_id in hybrid_doc_ids:
                results_dict['hybrid_hit_rate'] += 1
                results_dict['hybrid_mrr'] += 1/hybrid_doc_ids[doc_id]

            # if no hits, let's capture that
            else:
                results_dict['total_misses'] += 1
                
        except Exception as e:
            print(e)
            continue

    #use raw counts to calculate final scores
    calc_hit_rate_scores(results_dict, search_type=search_type)
    calc_mrr_scores(results_dict, search_type=search_type)
    
    end = time.perf_counter() - start
    print(f'Total Processing Time: {round(end/60, 2)} minutes')
    record_results(results_dict, chunk_size, './eval_results', as_text=True)
    return results_dict

def calc_hit_rate_scores(results_dict: Dict[str, Union[str, int]], 
                         search_type: Literal['kw', 'vector', 'hybrid', 'all']=['kw', 'vector']
                         ) -> None:
    if search_type == 'all':
        search_type = ['kw', 'vector', 'hybrid']
    for prefix in search_type:
        results_dict[f'{prefix}_hit_rate'] = round(results_dict[f'{prefix}_hit_rate']/results_dict['total_questions'],2)

def calc_mrr_scores(results_dict: Dict[str, Union[str, int]],
                    search_type: Literal['kw', 'vector', 'hybrid', 'all']=['kw', 'vector']
                    ) -> None:
    if search_type == 'all':
        search_type = ['kw', 'vector', 'hybrid']
    for prefix in search_type:
        results_dict[f'{prefix}_mrr'] = round(results_dict[f'{prefix}_mrr']/results_dict['total_questions'],2)

def create_dir(dir_path: str) -> None:
    '''
    Checks if directory exists, and creates new directory
    if it does not exist
    '''
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def record_results(results_dict: Dict[str, Union[str, int]], 
                   chunk_size: int, 
                   dir_outpath: str='./eval_results',
                   as_text: bool=False
                   ) -> None:
    '''
    Write results to output file in either txt or json format

    Args:
    -----
    results_dict: Dict[str, Union[str, int]]
        Dictionary containing results of evaluation
    chunk_size: int
        Size of text chunks in tokens
    dir_outpath: str
        Path to output directory.  Directory only, filename is hardcoded
        as part of this function.
    as_text: bool
        If True, write results as text file.  If False, write as json file.
    '''
    create_dir(dir_outpath)
    time_marker = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    ext = 'txt' if as_text else 'json'
    path = os.path.join(dir_outpath, f'retrieval_eval_{chunk_size}_{time_marker}.{ext}')
    if as_text:
        with open(path, 'a') as f:
            f.write(f"{results_dict}\n")
    else: 
        with open(path, 'w') as f:
            json.dump(results_dict, f, indent=4)

def add_params(client: WeaviateClient, 
               class_name: str, 
               results_dict: dict, 
               param_options: dict, 
               hnsw_config_keys: List[str]
              ) -> dict:
    hnsw_params = {k:v for k,v in client.show_class_config(class_name)['vectorIndexConfig'].items() if k in hnsw_config_keys}
    if hnsw_params:
        results_dict = {**results_dict, **hnsw_params}
    if param_options and isinstance(param_options, dict):
        results_dict = {**results_dict, **param_options}
    return results_dict
    




