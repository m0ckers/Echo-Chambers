def sentiment_bertweet(text):

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
    model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")

    # Define the sentiment labels
    sentiment_labels = ["Negative", "Neutral", "Positive"]

    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="pt"
    )

    # Perform the forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted sentiment label
    predicted_sentiment = torch.argmax(logits, dim=1).item()
    sentiment = sentiment_labels[predicted_sentiment]

    return sentiment, logits