
FROM jupyter/pyspark-notebook:spark-3.3.0

USER root

WORKDIR /app/project/dashboard

RUN pip install --no-cache-dir \
    streamlit==1.50.0 \
    numpy==1.23.4 \
    pandas==1.5.1 \
    scikit-learn==1.1.2 \
    joblib==1.2.0 \
    statsmodels==0.14.4 \
    plotly==5.24.1 \
    "supabase>=2.0.0,<3.0.0"

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501"]
