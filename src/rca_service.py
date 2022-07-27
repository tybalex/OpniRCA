from fastapi import FastAPI
from main_rca import main_rca

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to Opni RCA Service"}


@app.get("/get_root_cause")
async def get_root_cause():
    res = main_rca()
    return {"message":"results generated", "result": res}