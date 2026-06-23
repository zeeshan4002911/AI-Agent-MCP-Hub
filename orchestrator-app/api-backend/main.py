from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from agents import base
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()


@app.get("/")
def read_root():
    print("zz HUGGINGFACEHUB_API_TOKEN", os.environ.get("HUGGINGFACEHUB_API_TOKEN"))
    return {"message": "Welcome to FastAPI"}


@app.get("/sample-agent")
def sample_agent():
    res = base.sample_graph()
    return res
