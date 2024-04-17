import os.path
import re

import pytorch_lightning as pl
import torch
import wandb
from torch import optim
from torchmetrics import CharErrorRate
from torchmetrics.text import EditDistance
from torchmetrics.text import WordErrorRate
from torchtext.data.metrics import bleu_score

from utils import prepare_imgs
from utils import seed_torch, alternatives

editdistance = EditDistance()
cer = CharErrorRate()
wer = WordErrorRate()
seed_torch(42)


# cer_metric = load_metric("cer")
class Simple_Latex_OCR(pl.LightningModule):
    def __init__(self, cfg, init_model, processor, is_train=True):
        super().__init__()
        self.cfg = cfg
        self.is_train = is_train
        self.best_bleu = 0
        self.outputs = []
        self.train_loss = 0
        self.processor = processor
        self.eos_token = self.processor.tokenizer.eos_token
        self.bos_token = self.processor.tokenizer.bos_token
        self.pad_token = self.processor.tokenizer.pad_token
        self.model = init_model
        if self.is_train:
            self.processor.save_pretrained(self.cfg.dirpath)

    def configure_optimizers(self):
        # optimizer=optim.Adadelta(self.parameters(), lr=self.cfg.lr, weight_decay=1e-4)
        # optimizer = optim.AdamW(self.parameters(), lr=self.cfg.lr
        # #                         # , weight_decay=5e-2
        #                         )
        # scheduler = CustomLRScheduler(optimizer, steps_per_epoch=self.cfg.steps_per_epoch, epochs=self.cfg.epochs,
        #                                  initial_lr=self.cfg.lr)
        #
        # lr_scheduler = {'scheduler': scheduler,
        #                'interval': 'step',
        #                'frequency': 1
        #                }
        ##########################
        optimizer = optim.AdamW(self.parameters(), lr=self.cfg.lr
                                # , weight_decay=5e-2
                                )
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                                    patience=self.cfg.patience),
            'monitor': 'cer',
            'interval': 'epoch',
            'frequency': 1}
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_num):
        for k, v in batch.items():
            batch[k] = v.to(self.device)
        outputs = self.model(**batch)
        loss = outputs.loss
        self.train_loss += loss
        if (batch_num + 1) % 1 == 0:
            wandb.log({"train/train_loss": loss})
            self.log('loss', loss, on_step=True, prog_bar=True, logger=True)
            self.log('avg loss', self.train_loss / (batch_num + 1), on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_num):
        if self.trainer.current_epoch + 1 >= self.cfg.start_val_epoch:
            outputs = self.model.generate(batch["pixel_values"])
            x = batch["pixel_values"]
            truth = self.detokenize(batch["labels"])
            pred = self.detokenize(outputs)
            # print(f"pred:{pred} truth:{truth}")
            pred_str, label_str = self.compute_cer(pred_ids=outputs, label_ids=batch["labels"])
            bleu = bleu_score(pred, [alternatives(x) for x in truth])
            acc = sum([1 if label_str[i] == pred_str[i] else 0 for i in range(len(label_str))]) / x.size(0)

            cer_score = cer(pred_str, label_str)
            # cer = cer_metric.compute(predictions=pred_str, references=label_str)
            distance = editdistance(pred_str, label_str)
            word_error_rate = wer(pred_str, label_str)
            self.outputs.append({
                'cer': cer_score,
                'distance': distance,
                'word_error_rate': word_error_rate,
                'bleu': bleu,
                'acc': acc

            })

    def on_validation_epoch_end(self):
        self.train_loss = 0

        if self.trainer.current_epoch + 1 >= self.cfg.start_val_epoch:
            cer = sum([x['cer'] for x in self.outputs]) / len(self.outputs)
            distance = sum([x['distance'] for x in self.outputs]) / len(self.outputs)
            word_error_rate = sum([x['word_error_rate'] for x in self.outputs]) / len(self.outputs)
            bleu = sum([x['bleu'] for x in self.outputs]) / len(self.outputs)
            acc = sum([x['acc'] for x in self.outputs]) / len(self.outputs)

            print("\n")
            print(
                f"bleu：{bleu:.4f} acc：{acc:.4f} distance：{distance:.4f}  word_error_rate：{word_error_rate:.4f} cer：{cer:.4f} ")

            self.log('cer', cer)
            self.log('distance', distance)
            self.log('word_error_rate', word_error_rate)
            self.log('bleu', bleu)
            self.log('acc', acc)

            wandb.log({'val/cer': cer})
            wandb.log({'val/distance': distance})
            wandb.log({'val/word_error_rate': word_error_rate})
            wandb.log({'val/bleu': bleu})
            wandb.log({'val/acc': acc})

            if self.is_train and bleu >= self.best_bleu:
                self.best_bleu = bleu
                old_model = os.path.join(self.cfg.dirpath, "model.safetensors")
                if os.path.exists(old_model):
                    os.remove(old_model)
                self.model.save_pretrained(self.cfg.dirpath)


        else:
            self.log('cer', 1.0)
            self.log('distance', 100)
            self.log('word_error_rate', 1.0)
            self.log('bleu', 0.0)
            self.log('acc', 0.0)
        self.outputs = []

    def forward(self, img_list):
        pixel_values = self.processor(images=img_list, return_tensors="pt").pixel_values.to(self.model.device)
        outs = self.model.generate(
            pixel_values.to(self.device),
            return_dict_in_generate=True,
            output_scores=True,
        )
        logits = torch.stack(outs.scores, dim=1)
        scores = torch.softmax(logits, dim=-1).max(dim=2).values

        mean_probs = []
        for idx, example in enumerate(scores):
            cur_length = int(
                (outs.sequences[idx] != self.processor.tokenizer.pad_token_id).sum()
            )
            assert cur_length > 1
            # Obtain the geometric mean.
            # Note that the first element in example corresponds to the second element in sequence
            mean_probs.append(
                float((example[: cur_length - 1] + 1e-8).log().mean().exp())
            )

        generated_text = self.processor.batch_decode(
            outs.sequences, skip_special_tokens=True
        )
        assert len(img_list) == len(generated_text) == len(mean_probs)

        final_out = []
        for text, prob in zip(generated_text, mean_probs):
            final_out.append({'text': self.post_process(text), 'score': prob})
        return final_out

    @torch.no_grad()
    def recognize_formula(self, images,batch_size=1):
        input_imgs = prepare_imgs(images)
        results = []
        for i in range(0, len(input_imgs), batch_size):
            part_imgs = input_imgs[i: i + batch_size]
            results.extend(self(part_imgs))
        return results

    def detokenize(self, tokens):
        tokens = torch.where(tokens == -100, self.processor.tokenizer.pad_token_id, tokens)
        toks = [self.processor.tokenizer.convert_ids_to_tokens(tok) for tok in tokens]
        for b in range(len(toks)):
            for i in reversed(range(len(toks[b]))):
                if toks[b][i] is None:
                    toks[b][i] = ''
                toks[b][i] = toks[b][i].replace('Ġ', ' ').strip()
                if toks[b][i] in ([self.eos_token, self.bos_token, self.pad_token]):
                    del toks[b][i]
        # print(toks)
        return toks

    @staticmethod
    def post_process(s: str) -> str:
        """Remove unnecessary whitespace from LaTeX code.

        Args:
            s (str): Input string

        Returns:
            str: Processed image
        """
        text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
        letter = "[a-zA-Z]"
        noletter = r"[\W_^\d]"
        names = [x[0].replace(" ", "") for x in re.findall(text_reg, s)]
        s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
        news = s
        while True:
            s = news
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter), r"\1\2", s)
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter), r"\1\2", news)
            news = re.sub(r"(%s)\s+?(%s)" % (letter, noletter), r"\1\2", news)
            if news == s:
                break
            # Replace \eqno with ~~~.
        s = re.sub(r"\\eqno", "~~~", s)

        return s

    def compute_cer(self, pred_ids, label_ids):
        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)
        return pred_str, label_str


import math
from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, steps_per_epoch, epochs, initial_lr, last_epoch=-1, verbose=False):
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.initial_lr = initial_lr
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        current_epoch = self.last_epoch // self.steps_per_epoch
        current_step = self.last_epoch % self.steps_per_epoch
        steps = self.steps_per_epoch
        epochs = self.epochs
        initial_lr = self.initial_lr

        # Your custom LR logic
        if current_epoch < 1:
            new_lr = initial_lr / steps * (current_step + 1)
        elif 1 <= current_epoch <= int(epochs / 3 * 2):
            new_lr = 0.5 * (1 + math.cos((current_step + 1 + (current_epoch - 1) * steps) * math.pi / (
                    int(epochs / 3 * 2) * steps))) * initial_lr
        else:
            new_lr = 0.5 * (1 + math.cos(
                (current_step + 1 + (current_epoch - 1) * steps) * math.pi / (epochs * steps))) * initial_lr

        return [new_lr for _ in self.optimizer.param_groups]
