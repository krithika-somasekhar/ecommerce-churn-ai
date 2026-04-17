#!/bin/bash
# Start both the FastAPI server and Streamlit dashboard

# Start FastAPI in the background
uvicorn src.api:app --host 0.0.0.0 --port 8000 &

# Start Streamlit in the foreground
streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
