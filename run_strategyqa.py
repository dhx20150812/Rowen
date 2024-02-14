import os
import re
import json
from tqdm import tqdm


import asyncio
from google_serper import GoogleSerperAPIWrapper

from utils import *
from prompt_template import *


google_search = GoogleSerperAPIWrapper()


def remove_prefix(text: str) -> str:
    result = re.sub(r"^\d+\.\s", "", text)
    return result


def construct_input_message(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    return messages


def chain_of_thought_reasoning(question: str):
    messages = construct_input_message(
        STRATEGYQA_CHAIN_OF_THOUGHT_TEMPLATE.format(question=question)
    )
    # first query
    # long_answer = single_run(messages, model="gpt-4")
    long_answer = single_run(messages)
    messages.append({"role": "assistant", "content": long_answer})
    # second query
    messages.append(
        {
            "role": "user",
            "content": STRATEGYQA_FINAL_ANSWER_TEMPLATE.format(question=question),
        }
    )
    short_answer = single_run(messages)
    return long_answer, short_answer


def async_chain_of_thought_reasoning(questions, use_chinese=False):
    # first query
    messages = [
        construct_input_message(
            STRATEGYQA_CHAIN_OF_THOUGHT_TEMPLATE.format(question=question)
            if not use_chinese
            else STRATEGYQA_CHAIN_OF_THOUGHT_TEMPLATE_ZH.format(question=question)
        )
        for question in questions
    ]
    long_answers = asyncio.run(run_api(messages=messages))
    # long_answers = asyncio.run(run_api(messages=messages, model="gpt-4-0613"))
    messages = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": (
                    STRATEGYQA_CHAIN_OF_THOUGHT_TEMPLATE.format(question=question)
                    if not use_chinese
                    else STRATEGYQA_CHAIN_OF_THOUGHT_TEMPLATE_ZH.format(
                        question=question
                    )
                ),
            },
            {"role": "assistant", "content": long_answer},
            {
                "role": "user",
                "content": (
                    STRATEGYQA_FINAL_ANSWER_TEMPLATE.format(question=question)
                    if not use_chinese
                    else STRATEGYQA_FINAL_ANSWER_TEMPLATE_ZH.format(question=question)
                ),
            },
        ]
        for question, long_answer in zip(questions, long_answers)
    ]
    # second query
    short_answers = asyncio.run(run_api(messages=messages))
    return short_answers


def machine_translate(source_text, backward=False):
    # backward=True代表中文到英文
    prompt = (
        "Translate the following sentences into {}. Only response with translated sentence and do not include any useless content: \n".format(
            "Chinese" if not backward else "English"
        )
        + source_text
    )
    return single_run(messages=construct_input_message(prompt))


async def async_machine_translate(source_texts, backward=False):
    async def machine_translate(source_text, backward=False, retry=3):
        for _ in range(retry):
            try:
                prompt = (
                    "Translate the following sentences into {}. Only response with translated sentence and do not include any useless content: \n".format(
                        "Chinese" if not backward else "English"
                    )
                    + source_text
                )
                return single_run(messages=construct_input_message(prompt=prompt))
            except:
                await asyncio.sleep(10)
        return None

    responses = [
        machine_translate(source_text, backward) for source_text in source_texts
    ]
    return await asyncio.gather(*responses)


def evaluate_answer_factualness(eval_filepath=""):
    with open(eval_filepath, "r") as f:
        data = json.load(f)

    # compute the percentage of factual responses
    is_response_correct_list = [
        item["answer"] == is_supported(item["repaired_answer"].lower()) for item in data
    ]
    factual_response_percentage = (
        sum(is_response_correct_list) * 1.0 / len(is_response_correct_list)
    )
    print("Percentage of Factual Response: ", factual_response_percentage)

    assert len(is_response_correct_list) == len(data)
    for idx, item in enumerate(data):
        item["final_decision"] = is_response_correct_list[idx]

    file_name, file_extension = os.path.splitext(eval_filepath)
    with open(file_name + "_eval" + file_extension, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def reduce_hallucination_pipleline():
    k = 6
    threshold = 0.5

    with open("", "r") as f:
        data = json.load(f)

    generation_data = []

    for item in tqdm(data):
        item_copy = item.copy()

        try:
            question = item_copy["question"]

            (
                original_long_answer,
                original_short_answer,
            ) = chain_of_thought_reasoning(question)
            item_copy["original_long_answer"] = original_long_answer
            item_copy["original_short_answer"] = original_short_answer

            semantic_equivalent_queries_str = single_run(
                messages=construct_input_message(
                    prompt=SEMANTICALLY_EQUIVALENT_PERTERBATIONS_TEMPLATE.format(
                        question=question, k=k
                    )
                ),
                temperature=1.0,
            )
            perturbated_queries = [
                remove_prefix(query).strip()
                for query in semantic_equivalent_queries_str.split("\n")
            ]
            item_copy["perturbated_queries"] = perturbated_queries

            chinese_perturbated_queries = asyncio.run(
                async_machine_translate(perturbated_queries)
            )
            item_copy["chinese_perturbated_queries"] = chinese_perturbated_queries

            target_perturbated_answers = async_chain_of_thought_reasoning(
                perturbated_queries
            )
            item_copy["target_perturbated_answers"] = target_perturbated_answers

            verifier_perturbated_answers = async_chain_of_thought_reasoning(
                chinese_perturbated_queries, use_chinese=True
            )
            item_copy["verifier_perturbated_answers"] = verifier_perturbated_answers

            # Step 6. 验证中英文答案是否一致？
            consistency_check_results = [
                is_supported(en_answer.lower()) == is_supported_zh(zh_answer.strip())
                for en_answer, zh_answer in zip(
                    target_perturbated_answers, verifier_perturbated_answers
                )
            ]

            item_copy["valid_semantic_equivalent_QA_pair"] = [
                {
                    "target_query": query,
                    "target_answer": en_answer,
                    "verifier_query": zh_query,
                    "verifier_answer": zh_answer,
                    "is_consistent": consistency,
                }
                for query, en_answer, zh_query, zh_answer, consistency in zip(
                    perturbated_queries,
                    target_perturbated_answers,
                    chinese_perturbated_queries,
                    verifier_perturbated_answers,
                    consistency_check_results,
                )
            ]

            consistency_check_score = (
                sum(consistency_check_results) * 1.0 / len(consistency_check_results)
            )
            item_copy["consistency_check_score"] = consistency_check_score

            if consistency_check_score >= threshold:
                item_copy["repaired_answer"] = original_short_answer
                generation_data.append(item_copy)
                continue

            reformulated_queries = asyncio.run(
                run_api(
                    messages=[
                        construct_input_message(
                            prompt=STRATEGYQA_RETRIEVAL_REFORMULATION_TEMPLATE.format(
                                question=query
                            )
                        )
                        for query in perturbated_queries
                    ],
                    temperature=1.0,
                )
            )
            item_copy["reformulated_queries"] = reformulated_queries

            search_outputs = asyncio.run(google_search.run(reformulated_queries))
            retrieved_evidences = [
                query.rstrip("?") + "? " + output["content"]
                for query, result in zip(reformulated_queries, search_outputs)
                for output in result
            ]
            item_copy["retrieved_evidences"] = retrieved_evidences

            repaired_answer = single_run(
                messages=construct_input_message(
                    prompt=STRATEGYQA_REPAIR_HALLUCINATION_TEMPLATE.format(
                        question=question,
                        initial_long_answer=original_long_answer,
                        initial_short_answer=original_short_answer,
                        evidences="\n".join(retrieved_evidences),
                    )
                )
            )
            item_copy["repaired_answer"] = repaired_answer

            generation_data.append(item_copy)

        except:
            pass

    save_path = ""
    with open(
        save_path,
        "w",
    ) as f:
        json.dump(generation_data, f, indent=4, ensure_ascii=False)

    evaluate_answer_factualness(save_path)


if __name__ == "__main__":
    reduce_hallucination_pipleline()
