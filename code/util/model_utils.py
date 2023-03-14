def extract_text_activations(model, word_lists):
    activations = []
    for word in word_lists:
        text = clip.tokenize([word]).to(device)
        with torch.no_grad():
            activations.append(model.encode_text(text).data.numpy())
    return np.array(activations)