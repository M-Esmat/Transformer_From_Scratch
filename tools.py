import torch
import building_blocks as bb

# Inference
def translate(sentence, model, src_vocab, trg_vocab, device, max_length=20):
    model.eval()
    tokens = [src_vocab.get(word.lower(), 3) for word in sentence.split()]
    tokens = [1] + tokens + [2]
    tokens = tokens + [0]*(max_length - len(tokens))
    src = torch.tensor(tokens[:max_length], device=device).unsqueeze(0)
    
    trg = [1]
    for _ in range(max_length):
        trg_tensor = torch.tensor(trg, device=device).unsqueeze(0)
        with torch.no_grad():
            output = model(src, trg_tensor)
        
        pred_token = output.argmax(2)[:, -1].item()
        trg.append(pred_token)
        if pred_token == 2:  # EOS
            break
    
    inv_vocab = {v: k for k, v in trg_vocab.items()}
    return ' '.join([inv_vocab.get(idx, '<unk>') for idx in trg[1:-1]])

def load_test_model(checkpoint_path, TransformerClass, embed_size, device, max_length):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    test_model = TransformerClass(
        src_vocab_size=len(checkpoint['src_vocab']),
        trg_vocab_size=len(checkpoint['trg_vocab']),
        src_pad_idx=0,
        trg_pad_idx=0,
        embed_size=embed_size,
        num_layers=2,
        heads=4,
        forward_expansion=4,
        dropout=0.1,
        device=device,
        max_length=max_length
    ).to(device)

    test_model.load_state_dict(checkpoint['model_state_dict'])
    test_model.eval()

    print(f"✅ Model loaded successfully from {checkpoint_path}")
    
    return test_model, checkpoint['src_vocab'], checkpoint['trg_vocab']
