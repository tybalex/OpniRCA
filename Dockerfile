FROM python:3.8-slim

COPY ./src ./
RUN pip install --no-cache-dir -r ./requirements.txt

CMD ["uvicorn", "rca_service:app","--port"," 5050", "--reload"]

