# Deployment Guide

## Local
1. `docker-compose up -d --build`
2. API: `http://localhost:8000/docs`
3. Frontend: `http://localhost:3000`

## Kubernetes (baseline)
`kubectl apply -f infrastructure/kubernetes/deployment.yaml`
