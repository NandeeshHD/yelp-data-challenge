FROM jupyter/pyspark-notebook:d4cbf2f80a2a

# append spark path so that spark-submit is directly available 
ENV PATH=/usr/local/spark/bin:$PATH

# Spark ports
EXPOSE 4040-4042
# API port
EXPOSE 9001

# clone the project repo
RUN git -C $HOME clone https://github.com/NandeeshHD/yelp-data-challenge.git

# install python dependencies
RUN pip install --requirement $HOME/yelp-data-challenge/requirements.txt && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

USER root

# install cURL
RUN apt-get update && apt-get install -yq --no-install-recommends curl

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_UID