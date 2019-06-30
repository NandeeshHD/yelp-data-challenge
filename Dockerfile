FROM jupyter/pyspark-notebook:d4cbf2f80a2a

RUN export PATH="/usr/local/spark/bin:$PATH"
RUN git -C $HOME clone https://github.com/NandeeshHD/yelp-data-challenge.git
RUN pip install --requirement $HOME/yelp-data-challenge/requirements.txt && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER
