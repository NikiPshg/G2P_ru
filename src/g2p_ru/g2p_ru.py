import torch
from .transformer import TransformerBlock
from .tokenizer import Tokenizer
from .configs.config import config_g2p
import os
import string 

absolute = os.path.abspath(os.path.dirname(__file__))

class G2P_RU():
    def __init__(self):

        self.tokenizer = Tokenizer(dict_path=os.path.join(absolute, "./configs/ru_dict.json"))
        self.G2P = TransformerBlock(tokenizer=self.tokenizer, config=config_g2p)
        self.G2P.load_state_dict(torch.load(os.path.join(absolute,"./weight/wer2.pt")))

    def __call__(self, seq: str):
        words_and_punctuations = self._split_text(seq.lower())
        temp = []
        for item in words_and_punctuations:
            if item in string.punctuation:
                temp.append(item)
            else:
                phonemes = self.greedy_decode_grapheme(item, 32, self.tokenizer.sos_idx)
                temp.extend(phonemes + [' '])

        result = []
        for i, token in enumerate(temp):
            if token == ' ':
                if (i > 0 and temp[i - 1] in string.punctuation) or \
                (i < len(temp) - 1 and temp[i + 1] in string.punctuation) or \
                 temp[i - 1] in ' ':
                    continue  
            result.append(token)

        return result[:-1] if result[-1] == ' ' else result



    def greedy_decode_grapheme(self, src, max_len, start_token):
        with torch.no_grad():
            self.G2P.eval()
            enc_input_tokens = torch.tensor(self.tokenizer.encode(src))

            pad_id = [self.tokenizer.pad_idx]
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
                new_token = torch.ones(1, 1).long().fill_(next_word) 
                label = torch.cat([label, new_token], dim=1) 

                if next_word == self.tokenizer.eos_idx:
                    break

            pred = self.tokenizer.decode(label[0].tolist()[1:-1])
            return pred
        
    def _split_text(self, seq: str):
        result = []
        temp = ""

        for char in seq:
            if char in string.punctuation:
                if temp: 
                    result.append(temp)
                    temp = ""
                result.append(char) 
            elif char == " ":
                if temp: 
                    result.append(temp)
                    temp = ""
            else:
                temp += char

        if temp:  
            result.append(temp)

        if result and result[0] == " ":
            result.pop(0)
        if result and result[-1] == " ":
            result.pop()

        return result


if __name__ == '__main__':
    g2p_instance = G2P_RU()
    while True:
        inp = str(input())
        result = g2p_instance(inp) #['z', 'd', 'r', 'a', 's', 't', 'v', 'j', 'tj', 'e']
        print(result)
