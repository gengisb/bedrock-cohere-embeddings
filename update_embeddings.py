import json
import logging
import boto3
from botocore.exceptions import ClientError
from pymongo import MongoClient, UpdateOne
from datetime import datetime
import os
from typing import List, Dict, Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration
AWS_REGION = "us-west-2"
MODEL_ID = 'cohere.embed-multilingual-v3'
#MODEL_ID = 'cohere.embed-english-v3'
BATCH_SIZE = 96  # Cohere's maximum batch size http://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed.html
MONGODB_DATABASE = 'DATABASE'
MONGODB_COLLECTION = 'COLLECTION'
CLUSTER_ENDPOINT = 'CLUSTERNAME'
class MongoDBHandler:
    def __init__(self):
        self.client = self._get_connection()
        self.db = self.client[MONGODB_DATABASE]
        self.collection = self.db[MONGODB_COLLECTION]

    @staticmethod
    def _get_connection():
        mongodb_user = os.getenv('MONGODB_USER')
        mongodb_password = os.getenv('MONGODB_PASSWORD')
        connection_string = f"mongodb+srv://{mongodb_user}:{mongodb_password}@{CLUSTER_ENDPOINT}/"
        return MongoClient(connection_string)

    def get_documents_without_embeddings(self) -> List[Dict]:
        return list(self.collection.find(
            {},
            {"_id": 1, "text": 1}
        ))

    def update_embeddings_batch(self, documents: List[Dict], embeddings: List[List[float]]) -> int:
        operations = [
            UpdateOne(
                {"_id": doc['_id']},
                {"$set": {
                    "embedding": embedding
                }}
            )
            for doc, embedding in zip(documents, embeddings)
        ]

        if operations:
            result = self.collection.bulk_write(operations)
            return result.modified_count
        return 0

    def close(self):
        self.client.close()

class BedrockEmbeddings:
    def __init__(self):
        self.client = boto3.client(
            service_name='bedrock-runtime',
            region_name=AWS_REGION
        )

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        body = json.dumps({
            "texts": texts,
            "input_type": "search_document"
        })

        response = self.invoke_model(body)
        response_body = json.loads(response.get('body').read())
        return response_body.get('embeddings', [])

    def invoke_model(self, body: str) -> Dict[str, Any]:
        logger.info(f"Generating embeddings with Cohere model {MODEL_ID}")
        
        response = self.client.invoke_model(
            body=body,
            modelId=MODEL_ID,
            accept='*/*',
            contentType='application/json'
        )

        logger.info(f"Successfully generated embeddings with model {MODEL_ID}")
        return response

def process_documents_in_batches(mongo_handler: MongoDBHandler, 
                               bedrock_handler: BedrockEmbeddings) -> int:
    total_updated = 0
    documents = mongo_handler.get_documents_without_embeddings()
    total_documents = len(documents)
    
    logger.info(f"Found {total_documents} documents to process in {AWS_REGION}")

    for i in range(0, total_documents, BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]
        batch_size = len(batch)
        logger.info(f"Processing batch of {batch_size} documents")

        # Extract texts and generate embeddings
        texts = [doc.get('text', '') for doc in batch]
        embeddings = bedrock_handler.generate_embeddings(texts)

        # Update MongoDB
        batch_updated = mongo_handler.update_embeddings_batch(batch, embeddings)
        total_updated += batch_updated
        
        logger.info(f"Updated {batch_updated} documents in batch. "
                   f"Total updated: {total_updated}")
        logger.info(f"Completed batch {i//BATCH_SIZE + 1} of "
                   f"{(total_documents-1)//BATCH_SIZE + 1}")

    return total_updated

def main():
    mongo_handler = None
    try:
        mongo_handler = MongoDBHandler()
        bedrock_handler = BedrockEmbeddings()

        total_updated = process_documents_in_batches(mongo_handler, bedrock_handler)

        logger.info(f"Successfully updated {total_updated} documents with embeddings")
        print(f"Finished generating embeddings with Cohere model {MODEL_ID} "
              f"in region {AWS_REGION}")
        print(f"Total documents updated: {total_updated}")

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print(f"A client error occurred: {message}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
    finally:
        if mongo_handler:
            mongo_handler.close()

if __name__ == "__main__":
    main()
