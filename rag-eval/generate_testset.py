# generate_testset.py
from dotenv import load_dotenv
load_dotenv()

import os
import json
import instructor
from openai import AsyncOpenAI

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas.testset import TestsetGenerator
from ragas.llms.base import InstructorLLM, InstructorModelArgs
from ragas.embeddings import GoogleEmbeddings


def generate_testset(corpus_path: str, num_questions: int = 10) -> list[dict]:
    """
    Generates question-reference pairs from your document corpus.

    Args:
        corpus_path: Path to the plain text corpus file.
        num_questions: How many Q&A pairs to generate.

    Returns:
        List of dicts with keys: question, reference.
    """
    loader = TextLoader(corpus_path, encoding="utf-8")
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(raw_docs)
    print(f"Loaded {len(docs)} chunks.")

    raw_client = AsyncOpenAI(
        api_key=os.environ["GOOGLE_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    patched_client = instructor.from_openai(raw_client, mode=instructor.Mode.MD_JSON)
    llm = InstructorLLM(
        client=patched_client,
        model="gemini-2.5-flash",
        provider="openai",
        model_args=InstructorModelArgs(max_tokens=4096),
    )
    embeddings = GoogleEmbeddings(model="gemini-embedding-001")

    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)

    print(f"Generating {num_questions} test questions...")
    testset = generator.generate_with_langchain_docs(docs, testset_size=num_questions)

    output = []
    for sample in testset.samples:
        output.append({
            "question": sample.eval_sample.user_input,
            "reference": sample.eval_sample.reference,
        })
    return output


if __name__ == "__main__":
    generated = generate_testset("data/docs.txt", num_questions=10)
    with open("generated_testset.json", "w", encoding="utf-8") as f:
        json.dump(generated, f, indent=2, ensure_ascii=False)
    print(f"Generated {len(generated)} samples → generated_testset.json")