import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredFileLoader

class DataIngestion:
    def __init__(self, directory_path: str):
        self.directory_path = directory_path
        self.documents = []  # store documents here

        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)

            if filename.endswith(".txt"):
                loader = TextLoader(file_path)
            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                # fallback for other formats
                loader = UnstructuredFileLoader(file_path)

            docs = loader.load()
            self.documents.extend(docs)

    def get_documents(self):
        """Return the loaded documents"""
        return self.documents


if __name__ == "__main__":
    data_ingestion = DataIngestion(directory_path="./data")
    documents = data_ingestion.get_documents()
    print(documents)
