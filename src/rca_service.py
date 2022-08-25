# Third Party
from fastapi import FastAPI

# Local
from main_rca import main_rca
from utils import list_clusters_and_service

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to Opni RCA Service"}


@app.get("/get_root_cause/{cluster_id}")
async def get_root_cause(cluster_id: str):
    res = main_rca(cluster_id)
    return {"message": "results generated", "result": res}


@app.get("/get_clusters")
async def get_clusters():
    clusters = list_clusters_and_service()
    return {"message": f"active clusters : {list(clusters.keys())}"}
