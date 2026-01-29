

from langchain_text_splitters import RecursiveCharacterTextSplitter
from LoadData import DataIngestion


class Chunking:
    def split_documents(documents, chunk_size=1000, chunk_overlap=200):   
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        split_docs = text_splitter.split_documents(documents)
        return split_docs
    
if __name__ == "__main__":
    DataIngestionObj = DataIngestion(directory_path="./data")
    documents = DataIngestionObj.get_documents()
    chunked_documents = Chunking.split_documents(documents)
    print(f"Number of chunked documents: {len(chunked_documents)}")
    print(len(documents), len(chunked_documents))