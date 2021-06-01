def sequence_classification():
    from transformers import pipeline
    nlp = pipeline("sentiment-analysis")
    result = nlp("I hate you")[0]
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
    result = nlp("I love you")[0]
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")
    classes = ["not paraphrase", "is paraphrase"]
    sequence_0 = "The company HuggingFace is based in New York City"
    sequence_1 = "Apples are especially bad for your health"
    sequence_2 = "HuggingFace's headquarters are situated in Manhattan"
    paraphrase = tokenizer(sequence_0, sequence_2, return_tensors="pt")
    not_paraphrase = tokenizer(sequence_0, sequence_1, return_tensors="pt")
    paraphrase_classification_logits = model(**paraphrase).logits
    not_paraphrase_classification_logits = model(**not_paraphrase).logits
    paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
    not_paraphrase_results = torch.softmax(not_paraphrase_classification_logits, dim=1).tolist()[0]
    # Should be paraphrase
    for i in range(len(classes)):
        print(f"{classes[i]}: {int(round(paraphrase_results[i] * 100))}%")
    # Should not be paraphrase
    for i in range(len(classes)):
        print(f"{classes[i]}: {int(round(not_paraphrase_results[i] * 100))}%")


def mask_language_model():
    from transformers import pipeline
    nlp = pipeline("fill-mask")

    from pprint import pprint
    pprint(nlp(f"HuggingFace is creating a {nlp.tokenizer.mask_token} that the community uses to solve NLP tasks."))

    from transformers import AutoModelWithLMHead, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")
    sequence = f"Distilled models are smaller than the models they mimic. Using them instead of the large versions would help {tokenizer.mask_token} our carbon footprint."
    input = tokenizer.encode(sequence, return_tensors="pt")
    mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
    token_logits = model(input).logits
    mask_token_logits = token_logits[0, mask_token_index, :]
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

    for token in top_5_tokens:
        print(token)
        print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))

# text generation
def text_generation():
    from transformers import pipeline
    text_generator = pipeline("text-generation")
    print(text_generator("As far as I am concerned, I will", max_length=50, do_sample=False))

    from transformers import AutoModelWithLMHead, AutoTokenizer
    model = AutoModelWithLMHead.from_pretrained("xlnet-base-cased")
    tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
    # Padding text helps XLNet with short prompts - proposed by Aman Rusia in https://github.com/rusiaaman/XLNet-gen#methodology
    PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
    (except for Alexei and Maria) are discovered.
    The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
    remainder of the story. 1883 Western Siberia,
    a young Grigori Rasputin is asked by his father and a group of men to perform magic.
    Rasputin has a vision and denounces one of the men as a horse thief. Although his
    father initially slaps him for making such an accusation, Rasputin watches as the
    man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
    the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
    with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""
    prompt = "Today the weather is really nice and I am planning on "
    inputs = tokenizer.encode(PADDING_TEXT + prompt, add_special_tokens=False, return_tensors="pt")
    print(inputs)
    prompt_length = len(tokenizer.decode(inputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
    outputs = model.generate(inputs, max_length=250, do_sample=True, top_p=0.95, top_k=60)
    generated = prompt + tokenizer.decode(outputs[0])[prompt_length:]
    print(generated)

def name_entity():
    from transformers import pipeline
    nlp = pipeline("ner")
    sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very close to the Manhattan Bridge which is visible from the window."
    print(nlp(sequence))

    from transformers import AutoModelForTokenClassification, AutoTokenizer
    import torch
    model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    label_list = [
        "O",       # Outside of a named entity
        "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
        "I-MISC",  # Miscellaneous entity
        "B-PER",   # Beginning of a person's name right after another person's name
        "I-PER",   # Person's name
        "B-ORG",   # Beginning of an organisation right after another organisation
        "I-ORG",   # Organisation
        "B-LOC",   # Beginning of a location right after another location
        "I-LOC"    # Location
    ]
    sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very" \
            "close to the Manhattan Bridge."
    # Bit of a hack to get the tokens with the special tokens
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
    inputs = tokenizer.encode(sequence, return_tensors="pt")
    outputs = model(inputs).logits
    predictions = torch.argmax(outputs, dim=2)

def summary():
    from transformers import pipeline
    summarizer = pipeline("summarization")
    ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
    A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
    Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
    In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
    Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
    2010 marriage license application, according to court documents.
    Prosecutors said the marriages were part of an immigration scam.
    On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
    After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
    Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
    All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
    Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
    Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
    The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
    Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
    Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
    If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
    """
    print(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False))
    from transformers import AutoModelWithLMHead, AutoTokenizer
    model = AutoModelWithLMHead.from_pretrained("t5-small")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    # T5 uses a max_length of 512 so we cut the article to 512 tokens.
    inputs = tokenizer.encode("summarize: " + ARTICLE, return_tensors="pt", max_length=512)
    outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    print(outputs)

def translation():
    from transformers import pipeline
    translator = pipeline("translation_en_to_de")
    print(translator("Hugging Face is a technology company based in New York and Paris", max_length=40))

    from transformers import AutoModelWithLMHead, AutoTokenizer
    model = AutoModelWithLMHead.from_pretrained("t5-small")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    inputs = tokenizer.encode("translate English to German: Hugging Face is a technology company based in New York and Paris", return_tensors="pt")
    outputs = model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)
    print(outputs)

if __name__ == '__main__':
    sequence_classification()
    # text_generation()
    # name_entity()
    # summary()
    # translation()