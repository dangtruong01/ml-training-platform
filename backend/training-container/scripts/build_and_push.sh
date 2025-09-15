#!/bin/bash

# Build and push training container to Google Container Registry
set -e

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"ml-training-pipeline-sand-jjgq"}
IMAGE_NAME="ml-training"
TAG=${1:-"latest"}
FULL_IMAGE_NAME="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"

echo "🏗️ Building training container..."
echo "📋 Project: ${PROJECT_ID}"
echo "🐳 Image: ${FULL_IMAGE_NAME}"

# Navigate to the training-container directory
cd "$(dirname "$0")/.."

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t "${FULL_IMAGE_NAME}" .

# Push to Google Container Registry
echo "📤 Pushing to Google Container Registry..."
docker push "${FULL_IMAGE_NAME}"

echo "✅ Container built and pushed successfully!"
echo "🚀 Image URI: ${FULL_IMAGE_NAME}"

# Update the Vertex AI service with the new image URI
echo "💡 To use this image, update your CONTAINER_URI in the Vertex AI service:"
echo "   CONTAINER_URI=${FULL_IMAGE_NAME}"