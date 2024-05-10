import onnxruntime
import IPython.display as ipd
import torch
import commons
import utils
from text.symbols import symbols
from text import text_to_sequence
import numpy as np
from scipy.io.wavfile import write

ort_session = onnxruntime.InferenceSession("./onnx/tmp.onnx")
checkpoint_path = '/home/mugan/vits/models/G_251000.pth'
config_path = '/home/mugan/vits/models/ko_fine_tuning_config.json'
hps = utils.get_hparams_from_file(config_path)

def get_text(text):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    # text_norm = torch.LongTensor(text_norm)
    text_norm = np.array(text_norm, dtype=np.int64)
    return text_norm

txt = get_text("학습은 잘 마치셨나요? 좋은 결과가 있길 바래요.")
input_text = np.expand_dims(txt, axis=0)
input_text_len = np.array([txt.shape[0]], dtype=np.int64)

arg = {
    "text": input_text,
    "text_lengths": input_text_len,
}

# ONNX 런타임에서 계산된 결과값
ort_outs = ort_session.run(["output"], arg)
output = ort_outs[0]

write(f'result/onnx.wav', hps.data.sampling_rate, output)
ipd.display(ipd.Audio(output, rate=hps.data.sampling_rate, normalize=False))