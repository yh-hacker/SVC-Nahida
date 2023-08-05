import io
import os

import gradio as gr
import librosa
import numpy as np
import soundfile
from inference.infer_tool import Svc
import logging
import os
import paddle
import requests
import utils
from spleeter import Separator
import time  
from datetime import datetime, timedelta  

build_dir=os.getcwd()
if build_dir == "/home/aistudio":
    build_dir += "/build"

model_dir=build_dir+'/trained_models'

model_list_path = model_dir + "/model_list.txt"

current_time = datetime.now()  
one_hour_later = current_time + timedelta(hours=1)  

# 筛选出文件夹
models = []
for filename in os.listdir(model_dir):
    # 判断文件名是否以 '.pdparams' 结尾，并且不包含后缀部分
    if filename.endswith('.pdparams') and os.path.splitext(filename)[0].isalpha():
        models.append(os.path.splitext(filename)[0])
cache_model = {}

def callback(text):  
    if text == "reboot":  
        os._exit(0)
        current_time = datetime.now()  
        one_hour_later = current_time + timedelta(hours=1)  
    else:  
        global start_time  
        if time.time() - start_time >= 3600:
            os._exit(0)
            current_time = datetime.now()  
            one_hour_later = current_time + timedelta(hours=1)  
        else:
            return text  

def separate_fn(song_input):
    try:
        if song_input is None:
            return "请上传歌曲",None,None,None,None
        params_2stems = {
        'sample_rate': 44100,
        'frame_length': 4096,
        'frame_step': 1024,
        'T': 512,
        'F': 1024,
        'num_instruments': ['vocals', 'instrumental'],
        'output_dir': build_dir+'/output_2stems',
        'checkpoint_path': build_dir+'/spleeter',
        'use_elu': False}
        sampling_rate, song = song_input
        soundfile.write("temp.wav", song, sampling_rate, format="wav")
        # 初始化分离器
        sep = Separator(params_2stems)
        sep.separate('temp.wav')
        vocal_path = params_2stems["output_dir"]+"/temp-vocals.wav"
        instrumental_path = params_2stems["output_dir"]+"/temp-instrumental.wav"
        return "分离成功，请继续前往体验【转换】和【混音】",vocal_path,instrumental_path,vocal_path,instrumental_path
    except Exception as e:
        import traceback
        return traceback.format_exc() , None,None,None,None


def convert_fn(model_name, input_audio,input_audio_micro, vc_transform, auto_f0,cluster_ratio, slice_db, noise_scale):
    try:
        if model_name in cache_model:
            model = cache_model[model_name]
        else:
            if paddle.device.is_compiled_with_cuda()==False and len(cache_model)!=0:
                return f"目前运行环境为CPU，受制于平台算力，每次启动本项目只允许加载1个模型，当前已加载{next(iter(cache_model))}",None,None
            config_path = f"{build_dir}/trained_models/config.json"
            model = Svc(f"{build_dir}/trained_models/{model_name}.pdparams", config_path,mode="test")
            cache_model[model_name] = model
        if input_audio is None and input_audio_micro is None:
            return "请上传音频", None,None
        if input_audio_micro is not None:
            input_audio = input_audio_micro
        sampling_rate, audio = input_audio
        duration = audio.shape[0] / sampling_rate
        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        if sampling_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
        print(audio.shape)
        out_wav_path = "temp.wav"
        soundfile.write(out_wav_path, audio, 16000, format="wav")
        print(cluster_ratio, auto_f0, noise_scale)
        _audio = model.slice_inference(out_wav_path, model_name, vc_transform, slice_db, cluster_ratio, auto_f0, noise_scale)
        del model
        return "转换成功，请继续前往体验【混音】", (44100, _audio),(44100, _audio)
    except Exception as e:
        import traceback
        return traceback.format_exc() , None,None

def compose_fn(input_vocal,input_instrumental,mixing_ratio=0.5):
    try:
        outlog = "混音成功"
        if input_vocal is None:
            return "请上传人声", None
        if input_instrumental is None:
            return "请上传伴奏", None
        vocal_sampling_rate, vocal = input_vocal
        vocal_duration = vocal.shape[0] / vocal_sampling_rate
        vocal = (vocal / np.iinfo(vocal.dtype).max).astype(np.float32)
        if len(vocal.shape) > 1:
            vocal = librosa.to_mono(vocal.transpose(1, 0))
        if vocal_sampling_rate != 44100:
            vocal = librosa.resample(vocal, orig_sr=vocal_sampling_rate, target_sr=44100)

        instrumental_sampling_rate, instrumental = input_instrumental
        instrumental_duration = instrumental.shape[0] / instrumental_sampling_rate
        instrumental = (instrumental / np.iinfo(instrumental.dtype).max).astype(np.float32)
        if len(instrumental.shape) > 1:
            instrumental = librosa.to_mono(instrumental.transpose(1, 0))
        if instrumental_sampling_rate != 44100:
            instrumental = librosa.resample(instrumental, orig_sr=instrumental_sampling_rate, target_sr=44100)
        if len(vocal)!=len(instrumental):
            min_length = min(len(vocal),len(instrumental))
            instrumental = instrumental[:min_length]
            vocal = vocal[:min_length]
            outlog = "人声伴奏长度不一致，已自动截断较长的音频"

        mixed_audio = (1 - mixing_ratio) * vocal + mixing_ratio * instrumental
        mixed_audio_data = mixed_audio.astype(np.float32)
        return outlog,(44100,mixed_audio_data)
    except Exception as e:
        import traceback
        return traceback.format_exc() , None


app = gr.Blocks()

with app:
    gr.Markdown('<h1 style="text-align: center;">SVC歌声转换全流程体验（伴奏分离，转换，混音）</h1>')
    with gr.Tabs() as tabs:
        with gr.TabItem("人声伴奏分离"):
            gr.Markdown('<p>该项目人声分离的效果弱于UVR5，如自备分离好的伴奏和人声可跳过该步骤</p>')
            song_input = gr.Audio(label="上传歌曲（tips:上传后点击右上角✏可以进行歌曲剪辑）",interactive=True)
            gr.Examples(examples=[build_dir+"/examples/song/blue.wav",build_dir+"/examples/song/Counter_clockwise_Clock.wav",build_dir+"/examples/song/one_last_kiss.wav"],inputs=song_input,label="歌曲样例")

            btn_separate = gr.Button("人声伴奏分离", variant="primary")
            text_output1 = gr.Textbox(label="输出信息")
            vocal_output1 = gr.Audio(label="输出人声",interactive=False)
            instrumental_output1 = gr.Audio(label="输出伴奏",interactive=False)
        with gr.TabItem("转换"):
            model_name = gr.Dropdown(label="模型", choices=models, value="纳西妲")
            vocal_input1 = gr.Audio(label="上传人声",interactive=True)
            gr.Examples(examples=[build_dir+"/examples/vocals/blue_vocal.wav",build_dir+"/examples/vocals/Counter_clockwise_Clock_vocal.wav",build_dir+"/examples/vocals/one_last_kiss_vocal.wav"],inputs=vocal_input1,label="人声样例")
            btn_use_separate = gr.Button("使用【人声伴奏分离】分离的人声")
            micro_input = gr.Audio(label="麦克风输入（优先于上传的人声）",source="microphone",interactive=True)
            vc_transform = gr.Number(label="变调（半音数量,升八度12降八度-12）", value=0)
            cluster_ratio = gr.Number(label="聚类模型混合比例", value=0,visible=False)
            auto_f0 = gr.Checkbox(label="自动预测音高（转换歌声时不要打开，会严重跑调）", value=False)
            slice_db = gr.Number(label="静音分贝阈值（嘈杂的音频可以-30，干声保留呼吸可以-50）", value=-50)
            noise_scale = gr.Number(label="noise_scale", value=0.2)
            btn_convert = gr.Button("转换", variant="primary")
            text_output2 = gr.Textbox(label="输出信息")
            vc_output2 = gr.Audio(label="输出音频",interactive=False)

        with gr.TabItem("混音"):
            vocal_input2 = gr.Audio(label="上传人声",interactive=True)
            btn_use_convert = gr.Button("使用【转换】输出的人声")
            instrumental_input1 = gr.Audio(label="上传伴奏")
            gr.Examples(examples=[build_dir+"/examples/instrumental/blue_instrumental.wav",build_dir+"/examples/instrumental/Counter_clockwise_Clock_instrumental.wav",build_dir+"/examples/instrumental/one_last_kiss_instrumental.wav"],inputs=instrumental_input1,label="伴奏样例")
            btn_use_separate2 = gr.Button("使用【人声伴奏分离】分离的伴奏")
            mixing_ratio = gr.Slider(0, 1, value=0.75,step=0.01,label="混音比例（人声:伴奏）", info="人声:伴奏")
            btn_compose = gr.Button("混音", variant="primary")
            text_output3 = gr.Textbox(label="输出信息")
            song_output = gr.Audio(label="输出歌曲",interactive=False)

        with gr.TabItem("设置"):
            start_time = time.time()  
            output = gr.Textbox(label="输出",placeholder=f"距离下一次允许重启时间为{one_hour_later}")  
            btn_reboot = gr.Button("重启",variant="primary") 
        btn_separate.click(separate_fn, song_input, [text_output1, vocal_output1,instrumental_output1,vocal_input1,instrumental_input1])
        btn_convert.click(convert_fn, [model_name, vocal_input1,micro_input,vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale], [text_output2, vc_output2,vocal_input2])
        btn_compose.click(compose_fn,[vocal_input2,instrumental_input1,mixing_ratio],[text_output3,song_output])
        btn_reboot.click(callback,output)
        btn_use_convert.click(lambda x:x,vc_output2,vocal_input2)
        btn_use_separate.click(lambda x:x,vocal_output1,vocal_input1)
        btn_use_separate2.click(lambda x:x,instrumental_output1,instrumental_input1)

app.launch()
