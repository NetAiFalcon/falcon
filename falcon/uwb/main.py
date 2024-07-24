

from typing import Union
from fastapi import FastAPI

# FastAPI 애플리케이션 생성
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/info")
def read_item():
    with open("temp15.txt", 'r') as f:
        line = f.readline()
    return line

# uwb의 좌표를 보내는 api server