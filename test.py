import random
import string
import asyncio
import logging
import os
import json
import aiohttp
import openai
from openai import AsyncOpenAI

client = AsyncOpenAI()
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='caesar_cipher_test.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

# Add a console handler to display logs in the terminal as well
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

def caesar_shift(text, shift):
    result = ''
    for char in text:
        if char.isalpha():
            start = ord('a') if char.islower() else ord('A')
            result += chr((ord(char) - start + shift) % 26 + start)
        else:
            result += char
    return result

async def generate_phrase_with_api(length):
    logger.info(f"Generating a phrase of length {length} using Claude API")
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
        "anthropic-version": "2023-06-01"
    }
    data = {
        "model": "claude-3-5-sonnet-20240620",
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": f"Generate a random english phrase around {length} characters long. Only output the phrase nothing else. Example: the waves crashed against the beach"
            }
        ]
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status == 200:
                result = await response.json()
                content = result['content'][0]['text']
                print(f"Generated phrase: {content.strip()}")  # Print the generated phrase
                return content.strip()
            else:
                logger.error(f"API call failed. Status: {response.status}")
                print(f"API call failed. Status: {response.status}")  # Print the error
                return None

async def generate_phrases_with_api(num_phrases, length):
    logger.info(f"Generating {num_phrases} phrases of length {length} using Claude API")
    phrases = []
    tasks = [generate_phrase_with_api(length) for _ in range(num_phrases)]
    results = await asyncio.gather(*tasks)
    phrases = [phrase for phrase in results if phrase is not None]
    logger.info(f"Generated {len(phrases)} phrases")
    return phrases

def load_phrases_from_file(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return [line.strip() for line in f]
    return []

def save_phrases_to_file(filename, phrases):
    with open(filename, 'w') as f:
        for phrase in phrases:
            f.write(f"{phrase}\n")

def trim_lines(text):
    return '\n'.join([line.strip() for line in text.strip().split('\n')])

async def decipher_with_claude(encoded_phrase):
    logger.info(f"Attempting to decipher phrase using Claude API: {encoded_phrase}")
    print(encoded_phrase)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
        "anthropic-version": "2023-06-01"
    }
    data = {
        "model": "claude-3-5-sonnet-20240620",
        "max_tokens": 300,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": trim_lines(f"""
                    This text is encoded using a Caesar cipher: {encoded_phrase}
                    A Caesar cipher is a simple substitution cipher that shifts letters in the alphabet by a fixed number of positions.

                    For example, if we shift each letter in "lm M pszi csy" by 4 positions backward:
                    l -> h
                    m -> i
                    M -> I
                    p -> l
                    s -> o
                    z -> v
                    i -> e
                    c -> y
                    s -> o
                    y -> u
                    "lm M pszi csy" becomes "hi I love you."
                    ANSWER: hi I love you
                    Try to decode the given text using a similar method. What does it mean?

                    Provide your decoded answer after 'ANSWER:' on a new line. Keep exact matching punctuation. If you don't format your answer this way, it will fail the evaluation.
                """)
            }
        ]
    }

    for attempt in range(3):
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"API call completed. Status: {response.status}")
                    return result  # Return the entire result
                else:
                    error_message = await response.text()
                    logger.error(f"API call failed. Attempt {attempt + 1}. Status: {response.status}, Error: {error_message}")
                    if response.status != 500:
                        return None
    logger.error("API call failed after 3 attempts.")
    return None

async def decipher_with_openai(encoded_phrase):
    logger.info(f"Attempting to decipher phrase using OpenAI API: {encoded_phrase}")
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": trim_lines(f"""
                    This text is encoded using a Caesar cipher: {encoded_phrase}
                    A Caesar cipher is a simple substitution cipher that shifts letters in the alphabet by a fixed number of positions.

                    For example, if we shift each letter in "lm M pszi csy" by 4 positions backward:
                    l -> h
                    m -> i
                    M -> I
                    p -> l
                    s -> o
                    z -> v
                    i -> e
                    c -> y
                    s -> o
                    y -> u
                    "lm M pszi csy" becomes "hi I love you."
                    ANSWER: hi I love you
                    Try to decode the given text using a similar method. What does it mean?

                    Provide your decoded answer after 'ANSWER:' on a new line. Keep exact matching punctuation. If you don't format your answer this way, it will fail the evaluation.
                """)}
            ],
            max_tokens=300,
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        answer = content.split("ANSWER:")[-1].strip()
        return answer
    except Exception as e:
        logger.error(f"API call failed for phrase: {encoded_phrase[:20]}... Error: {str(e)}")
        return None

async def test_caesar_cipher():
    logger.info("Starting Caesar cipher test")
    phrases_file = 'test_phrases.txt'
    phrases = load_phrases_from_file(phrases_file)

    if len(phrases) < 100:
        logger.info(f"Only {len(phrases)} phrases found in file. Generating more...")
        new_phrases = await generate_phrases_with_api(100 - len(phrases), 50)
        phrases.extend(new_phrases)
        save_phrases_to_file(phrases_file, phrases)
    else:
        logger.info(f"Loaded {len(phrases)} phrases from file")

    logger.info(f"Using {len(phrases)} phrases for testing")

    results = {
        "claude": {shift: {"pass": 0, "fail": 0} for shift in range(1, 16)},
        "openai": {shift: {"pass": 0, "fail": 0} for shift in range(1, 16)}
    }

    async def test_phrase(shift, phrase, provider):
        encoded_phrase = caesar_shift(phrase, shift)
        if provider == "openai":
            full_response = await decipher_with_claude(encoded_phrase)
            deciphered_phrase = full_response['content'][0]['text'].split("ANSWER:")[-1].strip() if isinstance(full_response, dict) and 'content' in full_response else ""
        else:  # openai
            deciphered_phrase = await decipher_with_openai(encoded_phrase)

        provider_logger = logging.getLogger(f"{provider}_logger")
        if deciphered_phrase.lower() == phrase.lower():
            provider_logger.info(f"Test passed for {provider}, shift {shift}, phrase: {phrase[:20]}...")
            return "pass"
        else:
            provider_logger.error(f"Test failed for {provider}, shift {shift}, phrase: {phrase[:20]}...")
            provider_logger.error(f"Original: {phrase}")
            provider_logger.error(f"Encoded:  {encoded_phrase}")
            provider_logger.error(f"Deciphered: {deciphered_phrase}")
            return "fail"

    # for provider in ["claude", "openai"]:
    for provider in ["openai", "claude"]:
        provider_logger = logging.getLogger(f"{provider}_logger")
        file_handler = logging.FileHandler(f"{provider}_cipher_test.log", mode='w')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        provider_logger.addHandler(file_handler)

        provider_logger.info(f"Testing with {provider.capitalize()}")
        for shift in range(1, 16):
            provider_logger.info(f"Testing shift {shift}")
            test_phrases = random.sample(phrases, 10)
            tasks = [test_phrase(shift, phrase, provider) for phrase in test_phrases]
            shift_results = await asyncio.gather(*tasks)

            results[provider][shift]["pass"] = shift_results.count("pass")
            results[provider][shift]["fail"] = shift_results.count("fail")

            provider_logger.info(f"Completed testing for {provider}, shift {shift}")

        provider_logger.info("Caesar cipher test completed")
        provider_logger.info("Test Results:")
        for shift, result in results[provider].items():
            provider_logger.info(f"Shift {shift}: Passed {result['pass']}, Failed {result['fail']}")

        total_passed = sum(result['pass'] for result in results[provider].values())
        total_failed = sum(result['fail'] for result in results[provider].values())
        total_tests = total_passed + total_failed
        provider_logger.info(f"Total tests: {total_tests}")
        provider_logger.info(f"Total passed: {total_passed}")
        provider_logger.info(f"Total failed: {total_failed}")
        provider_logger.info(f"Success rate: {(total_passed / total_tests) * 100:.2f}%")

        provider_logger.removeHandler(file_handler)
        file_handler.close()

if __name__ == "__main__":
    try:
        asyncio.run(test_caesar_cipher())
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        # Explicitly close the event loop
        loop = asyncio.get_event_loop()
        loop.close()