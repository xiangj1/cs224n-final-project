import json
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient


class AzureNER:
    def __init__(self, key_path='./keys/Azure-NER-keys.json'):
        with open(key_path) as key_file:
            keys = json.load(key_file)
        key, endpoint = keys['key1'], keys['endpoint']

        credential = AzureKeyCredential(key)
        self.client = TextAnalyticsClient(
            endpoint=endpoint, credential=credential)

    def entity_recognition(self, document):
        if not document:
            raise ValueError("Document is empty or None")

        result = self.client.recognize_entities(documents=[document])[0]

        # Print named entities for debugging purposes
        # for entity in result.entities:
        #     print(f"Text: {entity.text}\tCategory: {entity.category}\tSubcategory: {entity.subcategory}\tScore: {entity.confidence_score:.2f}\tLength: {entity.length}\tOffset: {entity.offset}")

        return result.entities
