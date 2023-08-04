import paddle
import paddle.nn as nn
import paddle
import os
import numpy as np
import math
import paddle.nn as nn
import ffmpeg
from scipy.signal.windows import hann
from librosa.core import stft, istft

class UNet(nn.Layer):
    def __init__(self, use_elu=False):
        super(UNet, self).__init__()
        self.use_elu = use_elu
        self.pad  = nn.Pad2D(padding=[1, 2, 1, 2])

        ### Encoder ###
        # First Layer
        self.conv1     =  nn.Conv2D(2, 16, kernel_size=5, stride=2)   ## padding 
        self.encoder1  =  self.encoder_block(16)
        # Second Layer
        self.conv2     =  nn.Conv2D(16, 32, kernel_size=5, stride=2) 
        self.encoder2  =  self.encoder_block(32)
        # Third Layer
        self.conv3     =  nn.Conv2D(32, 64, kernel_size=5, stride=2)  
        self.encoder3  =  self.encoder_block(64)
        # Fourth Layer
        self.conv4     =  nn.Conv2D(64, 128, kernel_size=5, stride=2) 
        self.encoder4  =  self.encoder_block(128)
        # Fifth Layer
        self.conv5     =  nn.Conv2D(128, 256, kernel_size=5, stride=2) 
        self.encoder5  =  self.encoder_block(256)
        # Sixth Layer
        self.conv6     =  nn.Conv2D(256, 512, kernel_size=5, stride=2) 
        self.encoder6  =  self.encoder_block(512)

        ### Decoder ###
        # First Layer
        self.decoder1  =  self.decoder_block(512, 256, dropout=True)    
        # Second Layer
        self.decoder2  =  self.decoder_block(512, 128, dropout=True)
        # Third Layer
        self.decoder3  =  self.decoder_block(256, 64, dropout=True)
        # Fourth Layer
        self.decoder4  =  self.decoder_block(128, 32)
        # Fifth Layer
        self.decoder5  =  self.decoder_block(64, 16)
        # Sixth Layer
        self.decoder6  =  self.decoder_block(32, 1)

        # Last Layer
        self.mask      =  nn.Conv2D(1, 2, kernel_size=4, dilation=2, padding=3) 
        self.sig       =  nn.Sigmoid()

    def encoder_block(self, out_channel):
        if not self.use_elu:
            return nn.Sequential(
                nn.BatchNorm2D(out_channel, epsilon=1e-3, momentum=0.01),
                nn.LeakyReLU(0.2)
            )
        else:
            return nn.Sequential(
                nn.BatchNorm2D(out_channel, epsilon=1e-3, momentum=0.01),
                nn.ELU()
            )

    def decoder_block(self, in_channel, out_channel, dropout=False):
        layers = [
            nn.Conv2DTranspose(in_channel, out_channel, kernel_size=5, stride=2)
        ]
        if not self.use_elu:
            layers.append(nn.ReLU())
        else:
            layers.append(nn.ELU())
        layers.append(nn.BatchNorm2D(out_channel, epsilon=1e-3, momentum=0.01))
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        ### Encoder ###
        skip1   =  self.pad(x)
        skip1   =  self.conv1(skip1)
        down1   =  self.encoder1(skip1)
        
        skip2   =  self.pad(down1)
        skip2   =  self.conv2(skip2)
        down2   =  self.encoder2(skip2)

        skip3   =  self.pad(down2)
        skip3   =  self.conv3(skip3)
        down3   =  self.encoder3(skip3)
        
        skip4   =  self.pad(down3)
        skip4   =  self.conv4(skip4)
        down4   =  self.encoder4(skip4)

        skip5   =  self.pad(down4)
        skip5   =  self.conv5(skip5)
        down5   =  self.encoder5(skip5)

        skip6   =  self.pad(down5)
        skip6   =  self.conv6(skip6)
        down6   =  self.encoder6(skip6)

        ### Decoder ###
        up1     =  self.decoder1(skip6)
        up1     =  up1[:, :, 1: -2, 1: -2]   
        merge1  =  paddle.concat((skip5, up1), 1)
        
        up2     =  self.decoder2(merge1)
        up2     =  up2[:, :, 1: -2, 1: -2] 
        merge2  =  paddle.concat((skip4, up2), 1)

        up3     =  self.decoder3(merge2)
        up3     =  up3[:, :, 1: -2, 1: -2] 
        merge3  =  paddle.concat((skip3, up3), 1)

        up4     =  self.decoder4(merge3)
        up4     =  up4[:, :, 1: -2, 1: -2] 
        merge4  =  paddle.concat((skip2, up4), 1)

        up5     =  self.decoder5(merge4)
        up5     =  up5[:, :, 1: -2, 1: -2] 
        merge5  =  paddle.concat((skip1, up5), 1)

        up6     =  self.decoder6(merge5)
        up6     =  up6[:, :, 1: -2, 1: -2]
        
        m       =  self.mask(up6)
        
        m       =  self.sig(m)
        return m * x

class Separator(object):
    def __init__(self, params):
        self.num_instruments = params['num_instruments']
        self.output_dir = params['output_dir']
        self.model_list = nn.LayerList()

        for i, name in enumerate(self.num_instruments):
            print('Loading model for instrumment {}'.format(i))
            net = UNet(use_elu=params['use_elu'])
            net.eval()
            state_dict = paddle.load(os.path.join(params['checkpoint_path'], '%dstems_%s.pdparams' % (len(self.num_instruments), name)))
            net.set_dict(state_dict)
            self.model_list.append(net)

        self.T = params['T']
        self.F = params['F']
        self.frame_length = params['frame_length']
        self.frame_step = params['frame_step']
        self.samplerate = params['sample_rate']

    def _load_audio(
            self, path, offset=None, duration=None,
            sample_rate=None, dtype=np.float32):
        """ Loads the audio file denoted by the given path
        and returns it data as a waveform.

        :param path: Path of the audio file to load data from.
        :param offset: (Optional) Start offset to load from in seconds.
        :param duration: (Optional) Duration to load in seconds.
        :param sample_rate: (Optional) Sample rate to load audio with.
        :param dtype: (Optional) Numpy data type to use, default to float32.
        :returns: Loaded data a (waveform, sample_rate) tuple.
        :raise SpleeterError: If any error occurs while loading audio.
        """
        if not isinstance(path, str):
            path = path.decode()

        probe = ffmpeg.probe(path)

        metadata = next(
            stream
            for stream in probe['streams']
            if stream['codec_type'] == 'audio')
        n_channels = metadata['channels']
        if sample_rate is None:
            sample_rate = metadata['sample_rate']
        output_kwargs = {'format': 'f32le', 'ar': sample_rate}
        process = (
            ffmpeg
            .input(path)
            .output('pipe:', **output_kwargs)
            .run_async(pipe_stdout=True, pipe_stderr=True))
        buffer, _ = process.communicate()
        waveform = np.frombuffer(buffer, dtype='<f4').reshape(-1, n_channels)
        if not waveform.dtype == np.dtype(dtype):
            waveform = waveform.astype(dtype)
        return waveform, sample_rate

    def _to_ffmpeg_codec(codec):
        ffmpeg_codecs = {
            'm4a': 'aac',
            'ogg': 'libvorbis',
            'wma': 'wmav2',
        }
        return ffmpeg_codecs.get(codec) or codec

    def _save_to_file(
            self, path, data, sample_rate,
            codec=None, bitrate=None):
        """ Write waveform data to the file denoted by the given path
        using FFMPEG process.

        :param path: Path of the audio file to save data in.
        :param data: Waveform data to write.
        :param sample_rate: Sample rate to write file in.
        :param codec: (Optional) Writing codec to use.
        :param bitrate: (Optional) Bitrate of the written audio file.
        :raise IOError: If any error occurs while using FFMPEG to write data.
        """
        directory = os.path.dirname(path)
        #get_logger().debug('Writing file %s', path)
        input_kwargs = {'ar': sample_rate, 'ac': data.shape[1]}
        output_kwargs = {'ar': sample_rate, 'strict': '-2'}
        if bitrate:
            output_kwargs['audio_bitrate'] = bitrate
        if codec is not None and codec != 'wav':
            output_kwargs['codec'] = _to_ffmpeg_codec(codec)
        process = (
            ffmpeg
            .input('pipe:', format='f32le', **input_kwargs)
            .output(path, **output_kwargs)
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stderr=True, quiet=True))

        process.stdin.write(data.astype('<f4').tobytes())
        process.stdin.close()
        process.wait()

    def _stft(self, data, inverse=False, length=None):
        """
        Single entrypoint for both stft and istft. This computes stft and istft with librosa on stereo data. The two
        channels are processed separately and are concatenated together in the result. The expected input formats are:
        (n_samples, 2) for stft and (T, F, 2) for istft.
        :param data: np.array with either the waveform or the complex spectrogram depending on the parameter inverse
        :param inverse: should a stft or an istft be computed.
        :return: Stereo data as numpy array for the transform. The channels are stored in the last dimension
        """
        assert not (inverse and length is None)
        data = np.asfortranarray(data)
        N = self.frame_length
        H = self.frame_step
        win = hann(N, sym=False)
        fstft = istft if inverse else stft
        win_len_arg = {"win_length": None,
                       "length": length} if inverse else {"n_fft": N}
        n_channels = data.shape[-1]
        out = []
        for c in range(n_channels):
            d = data[:, :, c].T if inverse else data[:, c]
            s = fstft(d, hop_length=H, window=win, center=False, **win_len_arg)
            s = np.expand_dims(s.T, 2-inverse)
            out.append(s)
        if len(out) == 1:
            return out[0]
        return np.concatenate(out, axis=2-inverse)

    def _pad_and_partition(self, tensor, T):
        old_size = tensor.shape[3]
        new_size = math.ceil(old_size/T) * T
        tensor = nn.functional.pad(tensor, [0, new_size - old_size, 0, 0])
        split_size = new_size // T

        return paddle.concat(paddle.split(tensor, split_size, axis=3), axis=0)

    def separate(self, input_wav):
        wav_name = input_wav.split('/')[-1].split('.')[0]
        output_dir = self.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        source_audio, samplerate = self._load_audio(input_wav)  # Length * 2

        # assert int(samplerate) == 44100
        
        if source_audio.shape[1] == 1:
            source_audio = paddle.concat((source_audio, source_audio), axis=1)
        elif source_audio.shape[1] > 2:
            source_audio = source_audio[:, :2]

        stft = self._stft(source_audio)  # L * F * 2
        stft = stft[:, : self.F, :]

        stft_mag = abs(stft)  # L * F * 2
        stft_mag = paddle.to_tensor(stft_mag)
        stft_mag = stft_mag.unsqueeze(0).transpose([0, 3, 2, 1])  # 1 * 2 * F * L

        L = stft.shape[0]

        stft_mag = self._pad_and_partition(
            stft_mag, self.T)  # [(L + T) / T] * 2 * F * T
        stft_mag = stft_mag.transpose((0, 1, 3, 2))
        # stft_mag : B * 2 * T * F

        B = stft_mag.shape[0]
        masks = []

        stft_mag = stft_mag

        for model, name in zip(self.model_list, self.num_instruments):
            mask = model(stft_mag)
            masks.append(mask)
            paddle.save(model.state_dict(), '2stems_%s.pdparams' % name)

        mask_sum = sum([m ** 2 for m in masks])
        mask_sum += 1e-10

        for i in range(len(self.num_instruments)):
            mask = masks[i]
            mask = (mask ** 2 + 1e-10/2) / (mask_sum)
            mask = mask.transpose((0, 1, 3, 2)) # B x 2 X F x T
            mask = paddle.concat(paddle.split(mask, mask.shape[0], axis=0), axis=3)
            mask = mask.squeeze(0)[:, :, :L]  # 2 x F x L
            mask = mask.transpose([2, 1, 0])

            # End using GPU

            mask = mask.detach().numpy()

            stft_masked = stft * mask
            stft_masked = np.pad(
                stft_masked, ((0, 0), (0, 1025), (0, 0)), 'constant')

            wav_masked = self._stft(
                stft_masked, inverse=True, length=source_audio.shape[0])

            save_path = os.path.join(
                output_dir, (wav_name + '-' + self.num_instruments[i] + '.wav'))

            self._save_to_file(save_path, wav_masked,
                               samplerate, 'wav', '128k')

        print('Audio {} separated'.format(wav_name))