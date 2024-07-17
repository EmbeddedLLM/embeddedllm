import sys
import os
import time
import asyncio
from loguru import logger

# Add the 'src' directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the engine module
from embeddedllm import engine
from embeddedllm import sampling_params

async def benchmark(input_token_length, output_token_length, model_path, backend):
    log_file = f'profile_model_timing_{os.path.basename(model_path)}_{input_token_length}_{output_token_length}_{backend}.log'

    # Check if the log file already exists
    if not os.path.exists(log_file):
        logger.add(log_file)

    model = engine.EmbeddedLLMEngine(model_path, vision=False, device="cpu", backend=backend)
    logger.info(f"Model: {model_name}")

    model.tokenizer.chat_template = "{% for message in messages %}{{  message['content']}}{% endfor %}"  # Override

    prompt_text = """
    A large language model (LLM) is a computational model notable for its ability to achieve general-purpose language 
    generation and other natural language processing tasks such as classification. Based on language models, LLMs acquire 
    these abilities by learning statistical relationships from vast amounts of text during a computationally intensive 
    self-supervised and semi-supervised training process.[1] LLMs can be used for text generation, a form of generative AI, 
    by taking an input text and repeatedly predicting the next token or word.[2]

    LLMs are artificial neural networks that utilize the transformer architecture, invented in 2017. The largest and 
    most capable LLMs, as of June 2024, are built with a decoder-only transformer-based architecture, which enables 
    efficient processing and generation of large-scale text data.

    Historically, up to 2020, fine-tuning was the primary method used to adapt a model for specific tasks. However, 
    larger models such as GPT-3 have demonstrated the ability to achieve similar results through prompt engineering, 
    which involves crafting specific input prompts to guide the model's responses.[3] These models acquire knowledge 
    about syntax, semantics, and ontologies[4] inherent in human language corpora, but they also inherit inaccuracies 
    and biases present in the data they are trained on.[5]

    Some notable LLMs are OpenAI's GPT series of models (e.g., GPT-3.5 and GPT-4, used in ChatGPT and Microsoft Copilot), 
    Google's Gemini (the latter of which is currently used in the chatbot of the same name), Meta's LLaMA family of models, 
    Anthropic's Claude models, and Mistral AI's models.

    History
    Before 2017, there were a few language models that were large as compared to capacities then available. In the 1990s, 
    the IBM alignment models pioneered statistical language modelling. A smoothed n-gram model in 2001 trained on 0.3 
    billion words achieved then-SOTA perplexity.[6] In the 2000s, as Internet use became prevalent, some researchers 
    constructed Internet-scale language datasets ("web as corpus"[7]), upon which they trained statistical language 
    models.[8][9] In 2009, in most language processing tasks, statistical language models dominated over symbolic 
    language models, as they can usefully ingest large datasets.[10]

    After neural networks became dominant in image processing around 2012, they were applied to language modelling as 
    well. Google converted its translation service to Neural Machine Translation in 2016. As it was before Transformers, 
    it was done by seq2seq deep LSTM networks.


    An illustration of main components of the transformer model from the original paper, where layers were normalized 
    after (instead of before) multiheaded attention At the 2017 NeurIPS conference, Google researchers introduced the 
    transformer architecture in their landmark paper "Attention Is All You Need". This paper's goal was to improve upon 
    2014 Seq2seq technology,[11] and was based mainly on the attention mechanism developed by Bahdanau et al. in 2014.
    [12] The following year in 2018, BERT was introduced and quickly became "ubiquitous".[13] Though the original 
    transformer has both encoder and decoder blocks, BERT is an encoder-only model.

    Although decoder-only GPT-1 was introduced in 2018, it was GPT-2 in 2019 that caught widespread attention because 
    OpenAI at first deemed it too powerful to release publicly, out of fear of malicious use.[14] GPT-3 in 2020 went 
    a step further and as of 2024 is available only via API with no offering of downloading the model to execute locally. 
    But it was the 2022 consumer-facing browser-based ChatGPT that captured the imaginations of the general population 
    and caused some media hype and online buzz.[15] The 2023 GPT-4 was praised for its increased accuracy and as a 
    "holy grail" for its multimodal capabilities.[16] OpenAI did not reveal high-level architecture and the number 
    of parameters of GPT-4.

    Competing language models have for the most part been attempting to equal the GPT series, at least in terms of 
    number of parameters.[17]

    Since 2022, source-available models have been gaining popularity, especially at first with BLOOM and LLaMA, though 
    both have restrictions on the field of use. Mistral AI's models Mistral 7B and Mixtral 8x7b have the more permissive 
    Apache License. As of June 2024, The Instruction fine tuned variant of the Llama 3 70 billion parameter model is 
    the most powerful open LLM according to the LMSYS Chatbot Arena Leaderboard, being more powerful than GPT-3.5 but 
    not as powerful as GPT-4.[18]

    As of 2024, the largest and most capable models are all based on the Transformer architecture. Some recent 
    implementations are based on other architectures, such as recurrent neural network variants and Mamba 
    (a state space model).[19][20][21]

    Dataset preprocessing
    See also: List of datasets for machine-learning research § Internet
    Probabilistic tokenization
    Because machine learning algorithms process numbers rather than text, the text must be converted to numbers. 
    In the first step, a vocabulary is decided upon, then integer indexes are arbitrarily but uniquely assigned 
    to each vocabulary entry, and finally, an embedding is associated to the integer index. Algorithms include 
    byte-pair encoding and WordPiece.

    Probabilistic tokenization also compresses the datasets. Because LLMs generally require input to be an array 
    that is not jagged, the shorter texts must be "padded" until they match the length of the longest one. How many 
    tokens are, on average, needed per word depends on the language of the dataset.[22][23]

    BPE
    Using a modification of byte-pair encoding, in the first step, all unique characters (including blanks and 
    punctuation marks) are treated as an initial set of n-grams (i.e. initial set of uni-grams). Successively 
    the most frequent pair of adjacent characters is merged into a bi-gram and all instances of the pair are 
    replaced by it. All occurrences of adjacent pairs of (previously merged) n-grams that most frequently occur 
    together are then again merged into even lengthier n-gram repeatedly until a vocabulary of prescribed size 
    is obtained (in case of GPT-3, the size is 50257).[24] Token vocabulary consists of integers, spanning from 
    zero up to the size of the token vocabulary. New words can always be interpreted as combinations of the 
    tokens and the initial-set uni-grams.[25]

    A token vocabulary based on the frequencies extracted from mainly English corpora uses as few tokens as 
    possible for an average English word. An average word in another language encoded by such an English-optimized 
    tokenizer is however split into suboptimal amount of tokens. GPT-2 tokenizer can use up to 15 times more tokens 
    per word for some languages, for example for the Shan language from Myanmar. Even more widespread languages 
    such as Portuguese and German have "a premium of 50%" compared to English.[26]

    For example, here is how tokenizer used by GPT-3 (Legacy) split the following sentence tokenizer: texts -> 
    series of numerical "tokens".
    """
    input_tokens = model.tokenizer.encode(prompt_text)[:input_token_length-1]
    input_text = model.tokenizer.decode(input_tokens)
    print(input_text)
    input_tokens = model.tokenizer.encode(input_text)
    print(len(input_tokens))

    assert input_token_length-1 == len(input_tokens)

    PromptInputs = {
        "prompt": input_text
    }

    sampling_params_config = sampling_params.SamplingParams(
        max_tokens=output_token_length,
        top_p=0.1,
        top_k=1,
        temperature=1,
        repetition_penalty=0.01,
    )

    start = time.perf_counter()

    async def generate():
        results = []
        async for response in model.generate(
            inputs=PromptInputs,
            sampling_params=sampling_params_config,
            request_id="benchmark",
            stream=True,
        ):
            results.append(response)
        return results

    response = await generate()
    end = time.perf_counter()

    logger.info(response[0])  # Access the generated text from the response

    total_time_taken = end - start
    logger.info(f"Total time taken: {total_time_taken:.2f} seconds")

    average_tps = (input_token_length + output_token_length) / total_time_taken
    logger.info("Average tps: "+ str(average_tps))


token_ins = [128, 256, 512]
token_outs = [128, 256, 512]

model_path ="C:\\Users\\ryzzai\\Documents\\Phi-3-mini-4k-instruct-062024-int4\\onnx\\directml\\Phi-3-mini-4k-instruct-062024-int4"
model_name = os.path.basename(model_path)

backend = "cpu"

for input_token_length in token_ins:
    for output_token_length in token_outs:
        for i in range(50):
            # Run the async function using asyncio.run()
            asyncio.run(benchmark(input_token_length, output_token_length, model_path, backend))
