# Dockerfile for CREW agents
# Added by UCSD-WI26-CSE291A-AI-Agents-Project team
# Original CREW code licensed under Apache 2.0

FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN bash requirements.txt

CMD ["python", "algorithms/CAMON/__main__.py"]


# # How To Run?
# cd external/CREW/crew-algorithms/crew_algorithms/wildfire_alg

# # Build Docker image
# docker build -t crew-agents .

# # Run a CAMON agent
# docker run --env OPENAI_API_KEY=your_key_here crew-agents \
#     python algorithms/CAMON/__main__.py \
#     envs.level=Cut_Trees_Sparse_small \
#     envs.seed=483 envs.max_steps=20