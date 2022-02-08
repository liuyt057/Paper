from pydub import AudioSegment
import os
import sys

def trans_all_file(files_path, target="wav"):
    """
    批量转化音频音乐格式

    Args:
        files_path (str): 文件夹路径
        target (str, optional): 目标音乐格式. Defaults to "wav".
    """

    for filepath in os.listdir(files_path):
        # 路径处理
        modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
        datapath = os.path.join(modpath, files_path + filepath)

        # 分割为文件名字和后缀并载入文件
        input_audio = os.path.splitext(datapath)
        song = AudioSegment.from_file(datapath, input_audio[-1].split(".")[-1])

        # 导出
        song.export(f"{input_audio[0]}.{target}", format=target)

trans_all_file("D:\\Research\\水下数据集\\data\\test_wavdata\\1\\")