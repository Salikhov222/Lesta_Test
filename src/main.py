from typing import List
import uvicorn
import re
import math
from collections import Counter

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi import UploadFile
from fastapi.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory="templates")


def tokenize_text(text: str) -> List[str]:
    cleaned_text = re.sub(r'[^\w\s]', '', text)     # regular expression to clear text from characters
    words = cleaned_text.split()    # all words in file
    
    return words

def calculate_tf(words: List[str]) -> dict:
    tf = {}
    word_counts = Counter(words)    # number of occurrences
    for word in words:
        tf[word] = word_counts[word] / len(words)     # tf = number of occurrences of a word / word count
    
    return tf

def calculate_idf(files: List[UploadFile], tf: dict) -> list:
    num_documents = len(files)
    idf = {}
    word_count_per_file = {word: set() for word in tf}      # dict to keep track of files containing each word
    
    for file in files[1:]:
        contents = file.file.read()
        words = tokenize_text(contents.decode('utf-8').lower())
        for word in tf:
            if word in words:
                word_count_per_file[word].add(file.filename)

    for word, files_with_word in word_count_per_file.items():
        num_documents_with_word = len(files_with_word)
        idf_value = math.log(num_documents / (num_documents_with_word + 1))
        idf[word] = idf_value
    
    return idf

def get_top_50_words(tf: dict, idf: dict) -> List[dict]:
    top_50_words = []
    for word in sorted(idf, key=idf.get, reverse=True)[:50]:
        top_50_words.append({"word": word, "tf": tf[word], "idf": idf[word]})
    return top_50_words


@app.post("/analyze/")
async def upload_file(request: Request, files: List[UploadFile]):
    if not files:
        return {"message": "No files uploaded"}
    
    try: 
        contents = await files[0].read()
        if not contents:
            return {"message": "Uploaded file is empty"}
        
        words = tokenize_text(contents.decode('utf-8').lower())
        tf = calculate_tf(words)
        idf = calculate_idf(files, tf)
        top_50 = get_top_50_words(tf, idf)
    except Exception as e:
        return {"message": "There was an error uploading the file"}
    finally:
        for file in files:
            await file.close()
    return templates.TemplateResponse("result.html", {"request": request, "words": top_50, "idf": idf})


@app.get("/", response_class=HTMLResponse)
async def home():
    return FileResponse('templates/index.html')

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)

