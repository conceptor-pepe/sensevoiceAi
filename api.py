
import librosa
import numpy as np
import aiohttp
from fastapi import FastAPI, Form, UploadFile, HTTPException
from pydantic import HttpUrl, ValidationError, BaseModel, Field
from typing import List, Union
from funasr_onnx import SenseVoiceSmall
from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess
from io import BytesIO

DEVICE_ID = 5

class ApiResponse(BaseModel):
    message: str = Field(..., description="Status message indicating the success of the operation.")
    results: str = Field(..., description="Remove label output")
    label_result: str = Field(..., description="Default output")


app = FastAPI()
async def from_url_load_audio(audio: HttpUrl) -> BytesIO:
    """
    从URL下载音频文件
    
    Args:
        audio: 音频文件的URL
        
    Returns:
        BytesIO对象，包含下载的音频数据
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(
            audio,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0"
            },
        ) as response:
            if response.status != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download image: {response.status}",
                )
            image_bytes = await response.read()
            return BytesIO(image_bytes)

class BatchApiResponse(BaseModel):
    """批量API响应模型"""
    message: str = Field(..., description="操作状态消息")
    results: List[str] = Field(..., description="去除标签的输出列表")
    label_result: List[str] = Field(..., description="原始输出列表")

@app.post("/transcribe", response_model=Union[ApiResponse, BatchApiResponse])
async def upload_url(
    files: List[UploadFile] = Form(..., description="wav or mp3 audios in 16KHz"),
    keys: str = Form(..., description="name of each audio joined with comma"),
    lang: str = Form("auto", description="language of audio content")
):
    """
    处理批量音频文件的转录
    
    Args:
        files: 上传的音频文件列表，支持wav或mp3格式，采样率16KHz
        keys: 每个音频文件的名称，以逗号分隔的字符串
        lang: 音频内容的语言，默认为自动检测
        
    Returns:
        包含转录结果的响应
    """
    # 检查文件列表是否为空
    if not files:
        raise HTTPException(400, detail={"error": "没有提供有效的音频文件"})
    
    # 解析keys参数
    key_list = [key.strip() for key in keys.split(',') if key.strip()]
    if len(key_list) != len(files):
        raise HTTPException(400, detail={"error": "音频文件数量与keys参数数量不匹配"})
    
    results = []
    label_results = []
    
    # 处理每个文件
    for i, audio_file in enumerate(files):
        try:
            # 读取文件内容
            audio = BytesIO(await audio_file.read())
            # 使用指定的语言参数进行处理
            res = model(audio, language=lang, use_itn=True)
            results.append(rich_transcription_postprocess(res[0]))
            label_results.append(res[0])
        except Exception as e:
            # 记录单个文件处理失败，但继续处理其他文件
            results.append(f"处理文件 {key_list[i]} 时出错: {str(e)}")
            label_results.append(f"错误: {str(e)}")
    
    return {
        "message": f"成功处理 {len(files)} 个音频文件",
        "results": results,
        "label_result": label_results
    }


if __name__ == "__main__":

    model_dir = "iic/SenseVoiceSmall"
    device_id = DEVICE_ID  # Use GPU 0, automatically use CPU when not available
    batch_size = 16
    language = "auto"
    quantize = False # Quantization model, small size, fast speed, accuracy may be insufficient: model_quant.onnx

    def load_data(self, wav_content: Union[str, np.ndarray, List[str], BytesIO], fs: int = None) -> List:
        def load_wav(path: str) -> np.ndarray:
            waveform, _ = librosa.load(path, sr=fs)
            return waveform

        if isinstance(wav_content, np.ndarray):
            return [wav_content]

        if isinstance(wav_content, str):
            return [load_wav(wav_content)]

        if isinstance(wav_content, list):
            return [load_wav(path) for path in wav_content]
        
        if isinstance(wav_content, BytesIO):
            return [load_wav(wav_content)]
        
        raise TypeError(f"The type of {wav_content} is not in [str, np.ndarray, list]")
    
    SenseVoiceSmall.load_data = load_data

    model = SenseVoiceSmall(
        model_dir,
        quantize=quantize,
        device_id=device_id,
        batch_size=batch_size
        )

    print("\n\nDocs: http://127.0.0.1:8000/docs\n")
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)