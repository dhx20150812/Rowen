import asyncio
import re
import string
import time

import dashscope
import openai

openai.api_key = "YOUR OPENAI KEY"  # put your openai api key here
dashscope.api_key = "YOUR DASHSCOPE KEY"


def remove_prefix(text: str) -> str:
    result = re.sub(r"^\d+\.\s", "", text)
    return result


def single_run(messages, retry=3, model="gpt-3.5-turbo-0613", n=1, temperature=0.0):
    for _ in range(retry):
        try:
            output = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                n=n,
                temperature=temperature,
            )
            if n == 1:
                return output.choices[0].message.content.strip()
            else:
                return [choice.message.content for choice in output.choices]
        except:
            time.sleep(20)
    return None


def single_qwen_run(messages, model_name):
    for _ in range(3):
        try:
            response = dashscope.Generation.call(
                model=model_name,
                messages=messages,
                temperature=0.0,
                result_format="message",  # set the result to be "message" format.
            )
            return response["output"]["choices"][0]["message"]["content"]
        except:
            time.sleep(10)
    return None


async def run_completion_api(prompts):
    # Make api calls asynchronously
    async def single_run(prompt, retry=3):
        for _ in range(retry):
            try:
                output = openai.Completion.create(
                    model="gpt-judge/model/name",
                    prompt=prompt,
                    temperature=0,
                    max_tokens=1,
                    stop=None,
                    echo=False,
                    logprobs=2,
                )
                return output
            except:
                await asyncio.sleep(20)
        return None

    responses = [single_run(prompt) for prompt in prompts]
    return await asyncio.gather(*responses)


async def run_api(messages, model="gpt-3.5-turbo-0613", retry=3, temperature=0.0):
    # Make api calls asynchronously
    async def single_run(message, model, retry=3, temperature=0.0):
        for _ in range(retry):
            try:
                output = openai.ChatCompletion.create(
                    model=model,
                    messages=message,
                    temperature=temperature,
                )
                return output.choices[0].message.content.strip()
            except:
                await asyncio.sleep(20)
        return None

    responses = [single_run(message, model, retry, temperature) for message in messages]
    return await asyncio.gather(*responses)


def is_supported(generated_answer):
    generated_answer = generated_answer.lower()
    if "true" in generated_answer or "false" in generated_answer:
        if "true" in generated_answer and "false" not in generated_answer:
            is_supported = True
        elif "false" in generated_answer and "true" not in generated_answer:
            is_supported = False
        else:
            is_supported = generated_answer.index("true") > generated_answer.index(
                "false"
            )
    else:
        is_supported = all(
            [
                keyword
                not in generated_answer.lower()
                .translate(str.maketrans("", "", string.punctuation))
                .split()
                for keyword in [
                    "not",
                    "cannot",
                    "unknown",
                    "information",
                ]
            ]
        )
    return is_supported


def is_supported_zh(generated_answer):
    if "是" in generated_answer or "否" in generated_answer:
        if "是" in generated_answer and "否" not in generated_answer:
            is_supported = True
        elif "否" in generated_answer and "是" not in generated_answer:
            is_supported = False
        else:
            is_supported = generated_answer.index("是") > generated_answer.index("否")
    else:
        is_supported = all(
            [
                keyword
                not in generated_answer.lower()
                .translate(str.maketrans("", "", string.punctuation))
                .split()
                for keyword in [
                    "不",
                    "不能",
                    "未知",
                    "信息",
                ]
            ]
        )
    return is_supported
