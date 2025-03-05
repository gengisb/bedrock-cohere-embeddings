# Update Embeddings Script

This script generates and updates document embeddings in MongoDB using AWS Bedrock's Cohere embedding models.

## Overview

The script provides functionality to:
1. Fetch documents from MongoDB that need embeddings
2. Generate embeddings using AWS Bedrock's Cohere models
3. Update the MongoDB documents with the generated embeddings
   
## MongoDB Configuration
The script expects a MongoDB collection with documents containing a 'text' field. The generated embeddings will be stored in an 'embedding' field.

Example document structure:

`{
    "_id": ObjectId("..."),
    "text": "Document text content",
    "embedding": [...],  // Added by the script
}`
## Components

### MongoDBHandler Class
- Manages MongoDB connection and operations
- Methods:
  - `get_documents_without_embeddings()`: Retrieves documents that need embeddings
  - `update_embeddings_batch()`: Updates documents with their generated embeddings
  - `close()`: Closes the MongoDB connection

### BedrockEmbeddings Class
- Handles interaction with AWS Bedrock service
- Methods:
  - `generate_embeddings()`: Generates embeddings for a list of texts
  - `invoke_model()`: Makes the API call to AWS Bedrock

### Main Processing Functions
- `process_documents_in_batches()`: Processes documents in batches to generate and update embeddings
- `main()`: Entry point that orchestrates the embedding generation and update process

## Configuration

Key configuration parameters:
- `AWS_REGION`: "us-west-2"
- `MODEL_ID`: "cohere.embed-multilingual-v3" (or "cohere.embed-english-v3")
- `BATCH_SIZE`: 96 (Cohere's maximum batch size)
- MongoDB Configuration:
  - Database: 'input your DB'
  - Collection: 'input your Collection'
  - Cluster: 'input your Cluster'

## Prerequisites

Environment variables required:
- `MONGODB_USER`: MongoDB username
- `MONGODB_PASSWORD`: MongoDB password

Required Python packages:
- boto3
- pymongo
- botocore

## Usage

1. Set up the required environment variables:
```bash
export MONGODB_USER=your_username
export MONGODB_PASSWORD=your_password
```

2. Run the script:
```bash
python update_embeddings.py
```

The script will:
- Connect to MongoDB
- Find documents without embeddings
- Generate embeddings using AWS Bedrock
- Update the documents with their embeddings
- Process documents in batches of 96
- Log progress and results

## Error Handling

The script includes error handling for:
- AWS Bedrock client errors
- MongoDB connection issues
- General exceptions

All errors are logged using Python's logging module.

## Logging

The script provides detailed logging of:
- Number of documents being processed
- Batch progress
- Successful updates
- Any errors that occur during execution
