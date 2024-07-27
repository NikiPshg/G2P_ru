import torch
from transformer import TransformerBlock
from tokenizer import tokenizer
from config import config_g2p

G2P = TransformerBlock(tokenizer=tokenizer,
                       config=config_g2p)

G2P.load_state_dict(torch.load('wer2.pt'))

class g2p_ru:
    def __init__(self):  # Исправлено на __init__
        self.G2P = G2P

    def __call__(self, seq: str):
        seq = seq.lower()
        result = self.greedy_decode_grapheme(seq, 32, tokenizer.sos_idx)
        return result

    def greedy_decode_grapheme(self, src, max_len, start_token):
        with torch.no_grad():
            self.G2P.eval()
            enc_input_tokens = torch.tensor(tokenizer.encode(src))

            pad_id = [tokenizer.pad_idx]
            enc_num_padding_tokens = 32 - len(enc_input_tokens)

            encoder_input = torch.cat([
                enc_input_tokens,
                torch.tensor(pad_id * enc_num_padding_tokens)
            ], dim=0)

            encoder_mask = (encoder_input != pad_id[0]).unsqueeze(0).unsqueeze(0).int()

            src_mask = encoder_mask.unsqueeze(0)

            input_decoder = self.G2P.encode(encoder_input, src_mask)

            label = torch.zeros(1, 1).fill_(start_token).long()

            for _ in range(max_len - 1):
                tgt_mask = (torch.tril(torch.ones((label.size(1), label.size(1))))).unsqueeze(0)

                out = self.G2P.decode(input_decoder, src_mask, label, tgt_mask)
                prob = self.G2P.fc_out(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.item()

                # Обновляем label
                new_token = torch.ones(1, 1).long().fill_(next_word)  # Переносим на устройство
                label = torch.cat([label, new_token], dim=1)  # Объединяем тензоры на одном устройстве

                if next_word == tokenizer.eos_idx:
                    break
            pred = tokenizer.decode(label[0].tolist()[1:-1])

            return pred


if __name__ == '__main__':
    g2p_instance = g2p_ru()
    while True:
        inp = str(input())
        result = g2p_instance(inp) #['z', 'd', 'r', 'a', 's', 't', 'v', 'j', 'tj', 'e']
        print(result)
