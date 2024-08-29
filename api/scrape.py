from bs4 import BeautifulSoup
import requests
import textwrap
from fastapi import FastAPI, HTTPException
import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Index
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec

load_dotenv('.env.local')

app = FastAPI()

# Define the request model
class QueryRequest(BaseModel):
    query: str
#curl -X POST "http://localhost:8000/scrape/" -H "Content-Type: application/json" -d "{\"query\": \"https://www.ratemyprofessors.com/professor/1546494\" }"

@app.post("/scrape/")
async def perform_scrape(request: QueryRequest):
    try:
        link = request.query
        # Perform the page scrape
        page_scrape = requests.get(link)
        soup = BeautifulSoup(page_scrape.text, "html.parser")

        # Extract data
        school = soup.find("a", attrs={"href": "/school/66"})
        name = soup.find("div", attrs={"class": "NameTitle__Name-dowf0z-0 cfjPUG"})
        rating = soup.find("div", attrs={"class": "RatingValue__Numerator-qw8sqy-2 liyUjw"})
        difficulty = soup.find("div", attrs={"class": "FeedbackItem__FeedbackNumber-uof32n-1 kkESWs"})
        comments = soup.find_all("div", attrs={"class": "Comments__StyledComments-dzzyvm-0 gRjWel"})
        course = soup.find("b")

        # Handle missing elements safely
        school_text = school.text if school else "N/A"
        name_text = name.text if name else "N/A"
        rating_text = rating.text if rating else "N/A"
        difficulty_text = difficulty.text if difficulty else "N/A"
        course_text = course.text if course else "N/A"

        comments_arr = [textwrap.fill(comment.text, width=80) for comment in comments]

        data = {
            "professor": name_text,
            "college": school_text,
            "professorRating": rating_text,
            "classDifficulty": difficulty_text,
            "department": course_text,
            "reviews": comments_arr
        }

        # Process the embeddings
        await process_embeddings(data)

        # Return a response to the client
        return {"message": "Scraping and embedding process completed successfully", "data": data}

    except Exception as e:
        raise HTTPException(status_code=500, detail="Scrape was not successful.")

async def process_embeddings(data):
    # Initialize Pinecone client
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    

    # Initialize OpenAI client
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)

    #print(pinecone_api_key if pinecone_api_key else "nothing")
    #print(openai_api_key if openai_api_key else "nothing2")


    processed_data = []
    response = client.embeddings.create(
        input=data['reviews'],
        model="text-embedding-ada-002"  # Use a correct model name for embeddings
    )

    embedding = response.data[0].embedding
    processed_data.append({
        "values": embedding,
        "id": data["professor"],
        "metadata": {
            "college": data["college"],
            "review": data["reviews"],
            "department": data["department"],
            "rating": data["professorRating"],
            "difficulty": data["classDifficulty"]
        }
    })

    # Ensure the index is created in Pinecone with the correct settings
    index = pc.Index('rmp')
    index.upsert(vectors=processed_data, namespace="ns1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
