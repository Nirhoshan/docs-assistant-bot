import os
import re
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter


# Load OpenAI API key from .env file
load_dotenv(find_dotenv())

repo_path = "docs-choreo-dev/en/docs/"

headers_to_split_on = [
    ("#", "Header1"),
    ("##", "Header2"),
]
web_path = "https://wso2.com/choreo/docs/"
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)


def load_files(path):  # loads all the files
    files = []
    # ignoring all the files except the files related to the choreo console
    ignore_files = ['vs-code', 'index', 'page-not-found']
    for filename in os.listdir(path):
        child_path = os.path.join(path, filename)
        if os.path.isdir(child_path):
            if "choreo-cli" in child_path:
                continue
            files += load_files(child_path)
        elif filename.endswith(".md"):
            if any(ignore_file in filename for ignore_file in ignore_files):
                continue
            files.append((os.path.join(path, filename), open(os.path.join(path, filename))))
    return files


def text_to_anchor(text):  # Converts a text to anchor format to construct URL
    anchor = text.lower()
    anchor = anchor.replace(" ", "-")
    anchor = re.sub("[^0-9a-zA-Z-]", "", anchor)
    anchor = "#" + anchor
    return anchor


def chuck_docs(files):  # Chunks the retrieved files
    chunks = []
    for file in files:
        markdown_doc = file[1].read()
        chunked_doc = markdown_splitter.split_text(markdown_doc,)
        print(file[0])
        for chunk in chunked_doc:
            suffix = ""
            if "Header3" in chunk.metadata.keys():
                suffix = text_to_anchor(chunk.metadata["Header3"])
            elif "Header2" in chunk.metadata.keys():
                suffix = text_to_anchor(chunk.metadata["Header2"])
            file_name = file[0].replace("docs-choreo-dev/", "")
            chunk.metadata["filename"] = file_name
            chunk.metadata["doc_link"] = web_path+file[0][len(path):-3]+"/"+suffix
            chunk.page_content = chunk.page_content.replace("../../", f"{web_path}")
            chunk.page_content = chunk.page_content.replace("../", f"{web_path}")
            chunk.page_content = chunk.page_content.replace(".md", "")
            chunk.page_content = chunk.page_content.replace("{.cInlineImage-full}", "")
            header1_text = '#' + chunk.metadata["Header1"] if 'Header1' in chunk.metadata.keys() else ''
            header2_text = '\n##' + chunk.metadata["Header2"] if 'Header2' in chunk.metadata.keys() else ''
            header3_text = '\n###' + chunk.metadata["Header3"] if 'Header3' in chunk.metadata.keys() else ''
            content_text = '\n' + chunk.page_content
            chunk.page_content = f"{header1_text}{header2_text}{header3_text}{content_text}"
        chunks += chunked_doc
    return chunks


if __name__ == '__main__':
    print("Loading files...")
    docs = chuck_docs(load_files(repo_path))
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    vector_db = Milvus.from_documents(
        docs,
        embeddings,
        collection_name=os.getenv("COLLECTION_NAME"),
        drop_old=True,
        metadata_field="ChoreoMetadata",
        connection_args={
            "uri": os.getenv("ZILLIZ_CLOUD_URI"),
            "token": os.getenv("ZILLIZ_CLOUD_API_KEY"),
            "secure": True,
        },
    )
