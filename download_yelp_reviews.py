# imports
import os
import pandas as pd
import numpy as np
from serpapi import GoogleSearch
import uuid

# loop thru each page and get reviews
params = {
  "engine": "yelp_reviews",
  "place_id": "oBtORnu25mYpaS8eJQY_kQ",
  "api_key": "0c7b1581adbbd6d6ef87658d9dfc0936655dd13087442a0c23decb6c32645571",
  "hl": "en",
  "sortby": "date_desc"
}
start_list = np.arange(0, 650, 10) 
review_list = []
    
def main():
    
    # loop thru each page and call yelp_reviews API
    for start in start_list:
        params["start"] = start
        print(params)
        # call yelp review API
        search = GoogleSearch(params)
        results = search.get_dict()
        review_list.extend(results["reviews"])

if __name__ == "__main__":
    main()