import os
import csv
import logging
import unittest
import asyncio
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_recall, context_precision, faithfulness, answer_similarity
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from generate_response import app, lifespan, bulk_response, get_docs


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


async def initialize_app_state():
    async with lifespan(app):
        pass


async def process_questions(questions):
    await initialize_app_state()
    answers, contexts = [], []
    for question in questions:
        logger.info(question)
        answer = await bulk_response(question)
        logger.info(answer)
        docs = await get_docs(question)
        context = [(f"doc: {doc.page_content}. You can refer to this [document] "
                    f"({doc.metadata['ChoreoMetadata']['doc_link']}) for more details.") for doc in docs]
        answers.append(answer)
        contexts.append(context)
    return questions, answers, contexts


class AccuracyTest(unittest.TestCase):

    def test_metrics_thresholds(self):
        llm = ChatOpenAI(model="gpt-4-turbo-2024-04-09", temperature=1e-8)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        questions, ground_truths = load_data('test_data/validation_dataset.csv')
        questions, answers, contexts = asyncio.run(process_questions(questions))
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truths": ground_truths
        }
        dataset = Dataset.from_dict(data)
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
        results.to_csv('test_data/accuracy_results.csv', index=False)
        metric_columns = ['context_precision', 'context_recall', 'faithfulness', 'answer_similarity']
        metric_scores = results[metric_columns]
        mean_scores = metric_scores.mean()
        thresholds = {'context_precision': 0.95, 'context_recall': 0.90, 'faithfulness': 0.90,
                      'answer_similarity': 0.90}
        for metric, threshold in thresholds.items():
            with self.subTest(metric=metric):
                logger.info(f"{metric} average of {mean_scores[metric]:.2f} meets the threshold of {threshold}.")
                self.assertGreaterEqual(mean_scores[metric], threshold,
                                        f"{metric} average of {mean_scores[metric]:.2f} is below the threshold"
                                        f"of {threshold}.")


if __name__ == "__main__":
    unittest.main()
