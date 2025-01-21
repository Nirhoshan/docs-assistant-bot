import csv
import logging
import sys
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_recall, context_precision, faithfulness, answer_similarity
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from generate_response import get_docs, bulk_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

def load_data(filepath):
    questions, ground_truths = [], []
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            questions.append(row[0])
            ground_truths.append([row[1]])
    return questions, ground_truths

def process_questions(questions):
    answers, contexts = [], []
    for question in questions:
        answer = bulk_response(question)
        docs = get_docs(question)
        context = [(f"doc: {doc.page_content}. You can refer to this [document] "
                    f"({doc.metadata['ChoreoMetadata']['doc_link']}) for more details.") for doc in docs]
        answers.append(answer)
        contexts.append(context)
    return questions, answers, contexts

def main():
    # Initialize LLM and embeddings
    llm = ChatOpenAI(model="gpt-4-turbo-2024-04-09", temperature=1e-8)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Load and process data
    questions, ground_truths = load_data('test_data/validation_dataset.csv')
    questions, answers, contexts = process_questions(questions)

    # Prepare dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truths": ground_truths
    }
    dataset = Dataset.from_dict(data)

    # Run evaluation
    results = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_similarity,
        ],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False
    ).to_pandas()

    # Save results
    results.to_csv('test_data/accuracy_results.csv', index=False)

    # Define thresholds
    thresholds = {
        'context_precision': 0.95,
        'context_recall': 0.90,
        'faithfulness': 0.90,
        'answer_similarity': 0.90
    }

    # Check metrics against thresholds
    mean_scores = results[list(thresholds.keys())].mean()
    failed_metrics = []

    for metric, threshold in thresholds.items():
        score = mean_scores[metric]
        logger.info(f"{metric} average: {score:.2f} (threshold: {threshold})")
        if score < threshold:
            failed_metrics.append(f"{metric} ({score:.2f} < {threshold})")

    # Exit with error if any metrics failed
    if failed_metrics:
        logger.error("Accuracy test failed for the following metrics:")
        for metric in failed_metrics:
            logger.error(f"- {metric}")
        sys.exit(1)
    else:
        logger.info("All accuracy metrics passed!")

if __name__ == "__main__":
    main()