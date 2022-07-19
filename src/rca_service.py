from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to Opni RCA Service"}


@app.get("/get_root_cause")
async def get_root_cause():
    import subprocess

    command = "cd ..; bash main.sh"

    ret = subprocess.run(command, capture_output=True, shell=True)
    return {"message":"results generated", "info": ret}