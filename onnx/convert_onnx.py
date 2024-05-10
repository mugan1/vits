import IPython.display as ipd
import torch
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io.wavfile import write

checkpoint_path = '/home/mugan/vits/models/G_251000.pth'
config_path = '/home/mugan/vits/models/ko_fine_tuning_config.json'
destination_path = './onnx/tmp.onnx'

hps = utils.get_hparams_from_file(config_path)
spk_count = hps.data.n_speakers

model_g = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model)

model_g.eval()
utils.load_checkpoint(checkpoint_path, model_g, None)

def get_text(text):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

with torch.no_grad():
    model_g.dec.remove_weight_norm()
    
txt = get_text("학습은 잘 마치셨나요? 좋은 결과가 있길 바래요.")
input_text = txt.unsqueeze(0)
input_text_len = torch.LongTensor([txt.size(0)])
sid = torch.LongTensor([0])

def infer_forward(text, text_lengths):
    audio = model_g.infer(
        text,
        text_lengths,
        noise_scale=.667,
        length_scale=1,
        noise_scale_w=0.8,
        sid=torch.LongTensor([0]),
    )[0][0,0]

    return audio

model_g.forward = infer_forward
model_g.cpu()

output = infer_forward(input_text, input_text_len).data.float().numpy()

output_text = output
output_text_len = len(output)
write(f'result/test.wav', hps.data.sampling_rate, output)
ipd.display(ipd.Audio(output, rate=hps.data.sampling_rate, normalize=False))

arg = (input_text, input_text_len)

torch.onnx.export(model_g, arg, destination_path,
    verbose=False,
    opset_version=15,
    input_names=["text", "text_lengths"],
    output_names=["output"],
    dynamic_axes={
        "text": {0: "batch_size", 1: "phonemes"},
        "text_lengths": {0: "batch_size"},
        "output": {0: "batch_size", 1: "time"},
    },
)
