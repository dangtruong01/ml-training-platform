#!/bin/bash

# Build a fast, lightweight training container for testing
set -e

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"ml-training-pipeline-sand-jjgq"}
IMAGE_NAME="ml-training"
TAG="fast"
FULL_IMAGE_NAME="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"

echo "🚀 Building FAST training container..."
echo "📋 Project: ${PROJECT_ID}"
echo "🐳 Image: ${FULL_IMAGE_NAME}"
echo "⚡ Using lightweight CPU-only base image"

# Navigate to the training-container directory
cd "$(dirname "$0")/.."

# Build the Docker image using the fast Dockerfile
echo "🔨 Building Docker image..."
docker build -f Dockerfile.fast -t "${FULL_IMAGE_NAME}" .

# Push to Google Container Registry
echo "📤 Pushing to Google Container Registry..."
docker push "${FULL_IMAGE_NAME}"

echo "✅ Fast container built and pushed successfully!"
echo "🚀 Image URI: ${FULL_IMAGE_NAME}"
echo ""
echo "⚠️  NOTE: This is a CPU-only container for testing."
echo "   For GPU training, use the full container: ./scripts/build_and_push.sh"