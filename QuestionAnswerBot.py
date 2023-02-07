# imports
import os
import sys
import json
import time
import numpy as np
import uuid
import openai
from transformers import GPT2TokenizerFast
from serpapi import GoogleSearch

# tenacity import (or exponential backoff)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) 


class ReviewsProcessor(object):
    """ Class for downloading, processing, and generating embeddings of a Yelp reviews dataset. 
    
    Args:

    """
    def __init__(self,
                 engine,
                 place_id,
                 serp_api_key,
                 openai_api_key,
                 embedding_model="text-embedding-ada-002"):
        self.engine = engine
        self.place_id = place_id
        self.serp_api_key = serp_api_key
        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.reviews = None
        self.embeddings = None

    def download_reviews(self, start, end):
        
        # serp api parameters
        params = {
        "engine": self.engine,
        "place_id": self.place_id,
        "api_key": self.serp_api_key,
        "hl": "en",
        "sortby": "date_desc"
        }

        # get start and end range for the reviews 
        start_list = np.arange(start, end, 10) 
        review_list = []

        # loop thru each page and call yelp_reviews API
        for start in start_list:
            params["start"] = start
            # call yelp review API
            search = GoogleSearch(params)
            results = search.get_dict()
            review_list.extend(results["reviews"])
        
        # grab neccessary fields into another list
        self.reviews = {}
        for idx, review in enumerate(review_list):
            self.reviews[str(idx)] = {
                                "review_id": str(uuid.uuid4()),
                                "user_id": review["user"]["user_id"],
                                "address": review["user"]["address"],
                                "n_reviews": review["user"]["reviews"],
                                "comment": review["comment"]["text"],
                                "date": review["date"],
                                "rating": review["rating"],
                                "tokens": self.count_tokens(review["comment"]["text"])
                                }

    def compute_doc_embeddings(self):
        """
        Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
        
        Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
        """
        self.embeddings = {
            idx: self.get_doc_embedding(self.reviews[idx]["comment"].replace("\n", " ")) for idx in self.reviews
        }
        print("Computed doc embeddings.")

    def get_doc_embedding(self, text: str) -> list[float]:
        return self.get_embedding(text, self.embedding_model)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def get_embedding(self, text: str, model: str) -> list[float]:
        """
        Method that calls OpenAI embedding API to get embedding of text. 
        """
        result = openai.Embedding.create(
        model=model,
        input=text
        )
        return result["data"][0]["embedding"]

    def save_reviews(self, review_path):
        """ Method to save the downloaded and formatted reviews to JSON.
        """
        # save to a file
        save_path = os.path.join(review_path, f"{self.place_id}_reviews.json")
        with open(save_path, "w") as f:
            json.dump(self.reviews, f)
        print(f"Saved downloaded reviews to {review_path}.")

    def save_embeddings(self, embeddings_path):
        """ Method to save the downloaded and formatted reviews to JSON.
        """
        # save to a file
        save_path = os.path.join(embeddings_path, f"{self.place_id}_embeddings.json")
        with open(save_path, "w") as f:
            json.dump(self.embeddings, f)
        print(f"Saved downloaded reviews to {embeddings_path}.")

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a string"""
        return len(self.tokenizer.encode(text))
   

class Query(object):
    """ Question and answer bot class.

    Args:
        object (_type_): _description_
    """
    def __init__(self, 
                 place_id,
                 embedding_model="text-embedding-ada-002", 
                 completions_model="text-davinci-003",
                 max_section_len=500,
                 separator = "\n* ",
                 retry=True):
        self.place_id = place_id
        self.embedding_model = embedding_model
        self.completions_model = completions_model
        self.max_section_len = max_section_len
        self.separator = separator
        self.retry = retry
        self.embeddings = None
        self.reviews = None
        self.construct_separator()

    def answer_query(self, query: str, show_prompt: bool=False) -> str:
        """
        Method for asking the query.
        """
        if self.reviews is None or self.embeddings is None:
            raise Exception("Please load the reviews and embeddings first.")
    
        prompt = self.construct_prompt(query)
        
        if show_prompt:
            print(prompt["prompt"])

        COMPLETIONS_API_PARAMS = {    
            # We use temperature of 0.0 because it gives the most predictable, factual answer.
            "temperature": 0.0,
            "max_tokens": 300,
            "model": self.completions_model
            }

        response = openai.Completion.create(
                    prompt=prompt["prompt"],
                    **COMPLETIONS_API_PARAMS
                )

        # return response["choices"][0]["text"].strip(" \n")
        return {"response": response["choices"][0]["text"].strip(" \n"), "prompt": prompt}


    def construct_prompt(self, query: str) -> str:
        """
        Fetch relevant documents from the context and construct prompt. 
        """
        most_relevant_document_sections = self.order_document_sections_by_query_similarity(query)
        
        chosen_sections = []
        chosen_sections_len = 0
        chosen_sections_indexes = []
        
        for _, section_index in most_relevant_document_sections:
            # Add contexts until we run out of space.        
            document_section = self.reviews[str(section_index)]
            # document_section = df.loc[section_index]
            
            chosen_sections_len += document_section["tokens"] + self.separator_len
            if chosen_sections_len > self.max_section_len:
                break
                
            chosen_sections.append(self.separator + document_section["comment"].replace("\n", " "))
            chosen_sections_indexes.append(str(section_index))
                
        # Useful diagnostic information
        print(f"Selected {len(chosen_sections)} document sections:")
        print("\n".join(chosen_sections_indexes))
        
        header = ("Answer the question as truthfully as possible using the provided context,"
                   """ and if the answer is not contained within the text below, say "I don't know." """
                   "Also, output the reviews in the provided context that were most helpful when answering the question."
                   "\n\nContext:\n")

        return {"prompt": header + "".join(chosen_sections) + "\n\n Q: " + query + "\n A:", "context": "".join(chosen_sections), "header": header}
        # return header + "".join(chosen_sections) + "\n\n Q: " + query + "\n A:"
   
    def order_document_sections_by_query_similarity(self, query: str) -> list[(float, (str, str))]:
        """
        Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
        to find the most relevant sections. 
        
        Return the list of document sections, sorted by relevance in descending order.
        """
        query_embedding = self.get_query_embedding(query)
        
        document_similarities = sorted([
            (self.vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in self.embeddings.items()
        ], reverse=True)
        
        return document_similarities

    def construct_separator(self):
        """
        Creates separator for the class. Called when class is instantiated. 
        """
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.separator_len = len(tokenizer.tokenize(self.separator))

    def get_query_embedding(self, text: str) -> list[float]:
        """
        Method that calculates embedding given a query. 
        """
        return self.get_embedding(text, self.embedding_model)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def get_embedding(self, text: str, model: str) -> list[float]:
        """
        Method that calls OpenAI embedding API to get embedding of text. 
        """
        result = openai.Embedding.create(
        model=model,
        input=text
        )
        return result["data"][0]["embedding"]

    def vector_similarity(self, x: list[float], y: list[float]) -> float:
        """
        We could use cosine similarity or dot product to calculate the similarity between vectors.
        In practice, we have found it makes little difference. 
        """
        return np.dot(np.array(x), np.array(y))   

    def load_data(self, embedding_path: str, review_path: str):
        """
        Method for loading necessary data into the class instance.
        """
        self.__load_embeddings(embedding_path=embedding_path)
        self.__load_reviews(review_path=review_path)

    def __load_embeddings(self, embedding_path: str):
        """
        Method for loadding embeddings into the class instance.
        """
        with open(embedding_path, "r") as f:
            self.embeddings = json.load(f)
            print(f"Loaded {len(self.embeddings.keys())} embeddings.")

    def __load_reviews(self, review_path: str):
        """
        Method for loadding reviews into the class instance.
        """
        with open(review_path, "r") as f:
            self.reviews = json.load(f)
            print(f"Loaded {len(self.reviews.keys())} reviews.")
