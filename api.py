import librosa
import numpy as np
# import aiohttp # aiohttp 在当前代码片段中未被使用，如果后续需要可以取消注释
from fastapi import FastAPI, Form, UploadFile, HTTPException
from pydantic import HttpUrl, ValidationError, BaseModel, Field
from typing import List, Union
from funasr_onnx import SenseVoiceSmall
from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess
from io import BytesIO
import asyncio # 导入 asyncio 库，用于异步编程
import functools # 导入 functools 库，用于 functools.partial
import concurrent.futures # 导入 concurrent.futures 模块，用于创建线程池

DEVICE_ID = 5 # DEVICE_ID: 指定使用的计算设备ID，例如GPU的编号
MODEL_WORKERS = 1 # MODEL_WORKERS: 为GPU推理任务配置的工作线程数，通常为1以避免GPU争用

# --- Pydantic 模型定义 ---
class ApiResponse(BaseModel):
    """单文件API响应模型 (当前未使用，但保留定义以备将来扩展)"""
    message: str = Field(..., description="状态消息，指示操作的成功。") # message: 操作状态消息
    results: str = Field(..., description="去除标签的输出。") # results: 去除时间戳和标点等标签后的文本结果
    label_result: str = Field(..., description="默认输出。") # label_result: 原始的、带标签的文本结果

class BatchApiResponse(BaseModel):
    """批量API响应模型"""
    message: str = Field(..., description="操作状态消息，可能包含成功和失败的计数。") # message: 操作状态消息，例如 "成功处理 X 个文件中的 Y 个，失败 Z 个"
    results: List[str] = Field(..., description="去除标签的输出结果列表，每个元素对应一个文件。") # results: 存储所有文件去除标签后的转录文本列表
    label_result: List[str] = Field(..., description="原始输出结果列表，每个元素对应一个文件。") # label_result: 存储所有文件原始的、带标签的转录文本列表

# --- SenseVoiceSmall 模型加载和猴子补丁 ---
# 将 load_data 定义在全局作用域，以便模型初始化时使用
def load_data(self, wav_content: Union[str, np.ndarray, List[str], BytesIO], fs: int = None) -> List:
    """
    自定义数据加载函数，用于SenseVoiceSmall模型。
    这个函数会被用来替换SenseVoiceSmall类中原有的load_data方法（猴子补丁）。

    Args:
        self: SenseVoiceSmall模型实例。
        wav_content (Union[str, np.ndarray, List[str], BytesIO]): 音频内容。
                     可以是文件路径字符串，NumPy数组，文件路径列表，或BytesIO对象。
        fs (int, optional): 目标采样率。如果提供，音频将被重采样到此采样率。默认为None。

    Returns:
        List: 包含一个或多个NumPy数组的列表，每个数组代表一个音频波形。
    """
    # load_wav: 内部辅助函数，用于从路径加载和重采样单个WAV文件
    def load_wav(path_or_buffer) -> np.ndarray: # path_or_buffer: 音频文件路径或BytesIO对象
        # waveform: 加载后的音频波形数据 (NumPy数组)
        # _: librosa.load 返回的采样率，这里我们不直接使用它，因为模型通常有自己的处理方式或期望的采样率
        waveform, _ = librosa.load(path_or_buffer, sr=fs)
        return waveform

    # 根据 wav_content 的类型进行处理
    if isinstance(wav_content, np.ndarray):
        # wav_content: NumPy数组形式的音频波形
        return [wav_content] # 如果已经是NumPy数组，直接返回列表包含该数组

    if isinstance(wav_content, str):
        # wav_content: 字符串形式的音频文件路径
        return [load_wav(wav_content)] # 如果是字符串（路径），加载并返回

    if isinstance(wav_content, list):
        # wav_content: 包含多个音频文件路径的列表
        # [load_wav(path) for path in wav_content]: 对列表中的每个路径调用load_wav，并收集结果
        return [load_wav(path) for path in wav_content] # 如果是列表（路径列表），逐个加载并返回
    
    if isinstance(wav_content, BytesIO):
        # wav_content: BytesIO对象形式的音频数据
        return [load_wav(wav_content)] # 如果是BytesIO对象，加载并返回
    
    # 如果 wav_content 类型不支持，则抛出类型错误
    raise TypeError(f"不支持的音频内容类型: {type(wav_content)}。可接受类型为 [str, np.ndarray, list, BytesIO]")

# 应用猴子补丁：将自定义的 load_data 函数设置为 SenseVoiceSmall 类的方法
SenseVoiceSmall.load_data = load_data

# 初始化模型实例 (移至全局作用域)
# model_dir: 模型文件所在的目录路径
model_dir = "iic/SenseVoiceSmall"
# batch_size: 模型推理时使用的批处理大小
batch_size = 16
# quantize: 是否使用量化模型。量化模型体积小，速度快，但精度可能略有下降。
quantize = False 

# model: SenseVoiceSmall 模型的全局实例
model = SenseVoiceSmall(
    model_dir,
    quantize=quantize,
    device_id=DEVICE_ID, # device_id: 指定推理设备
    batch_size=batch_size
)

# --- 全局线程池执行器 ---
# model_executor: 初始化一个全局的线程池执行器，专门用于处理模型推理等阻塞型CPU/GPU密集任务
# max_workers=MODEL_WORKERS: 限制了同时执行模型推理的线程数量，对于GPU任务，通常设为1
model_executor = concurrent.futures.ThreadPoolExecutor(max_workers=MODEL_WORKERS)

# --- FastAPI 应用实例 ---
app = FastAPI() # app: FastAPI 应用的主实例

@app.on_event("shutdown")
async def app_shutdown():
    """
    应用程序关闭时调用的事件处理器。
    负责优雅地关闭全局模型推理线程池。
    """
    global model_executor # 引用全局执行器实例
    if model_executor:
        print("正在关闭模型推理线程池...") # 提示：正在关闭线程池
        # model_executor.shutdown(wait=True) 会等待所有已提交的任务完成后再关闭线程池。
        # 这对于确保所有正在进行的推理任务都能完成非常重要。
        model_executor.shutdown(wait=True)
        print("模型推理线程池已关闭。") # 提示：线程池已关闭

# --- 异步辅助函数：处理单个音频文件 ---
async def _process_audio_file(
    audio_file: UploadFile, # audio_file: 代表上传的单个音频文件的对象
    key: str,               # key: 与该音频文件关联的唯一标识符或名称
    lang: str,              # lang: 音频内容的语言代码 (例如 "auto", "zh", "en")
    model_instance: SenseVoiceSmall # model_instance: SenseVoiceSmall 模型的实例
) -> Union[tuple, Exception]:
    """
    异步处理单个音频文件。
    模型推理部分将在专门的线程池中执行，以避免阻塞事件循环。
    """
    try:
        # audio_content: 从 UploadFile 异步读取的音频文件字节内容
        audio_content: bytes = await audio_file.read()
        # audio_fp: 将字节内容包装成的 BytesIO 对象，方便模型读取和处理
        audio_fp = BytesIO(audio_content)
        
        # 获取当前 asyncio 事件循环实例
        loop = asyncio.get_event_loop()

        # model_func_partial: 创建一个偏函数，预设 model_instance 的 language 和 use_itn 参数。
        model_func_partial = functools.partial(model_instance, language=lang, use_itn=True)
        
        # res: 模型对音频文件进行语音识别的原始结果。
        # 使用 loop.run_in_executor 将阻塞的 model_func_partial(audio_fp) 调用
        # 放入我们定义的 model_executor 线程池中执行。
        res = await loop.run_in_executor(
            model_executor,     # executor: 使用全局定义的、固定大小的线程池
            model_func_partial, # func: 要在线程池中执行的函数 (已绑定参数的偏函数)
            audio_fp            # *args: 传递给 func 的位置参数 (这里是音频数据)
        )
        
        # processed_text: 对原始识别结果 res[0] 进行后处理（例如，去除标签、规范化文本格式）后的文本。
        processed_text: str = rich_transcription_postprocess(res[0])
        # raw_text: 未经 rich_transcription_postprocess 处理的原始模型输出。
        raw_text = res[0] # 通常 res[0] 直接就是原始的带标签文本或包含更丰富信息的结构

        return processed_text, raw_text # 返回成功处理的结果
    except Exception as e:
        # e: 在处理此文件期间捕获到的任何异常对象
        # print(f"处理文件 {key} 时发生错误: {e}") # 简单打印错误，实际应用中应使用日志库
        return e

# --- API 端点 ---
@app.post("/transcribe", response_model=BatchApiResponse) # 明确指定响应模型为 BatchApiResponse
async def transcribeHandler(
    files: List[UploadFile] = Form(..., description="16KHz采样率的WAV或MP3格式音频文件列表"), # files: 用户上传的音频文件列表
    keys: str = Form(..., description="每个音频文件的名称（标识符），以逗号分隔"), # keys: 与文件列表对应的名称字符串
    lang: str = Form("auto", description="音频内容的语言代码（例如 'auto', 'zh', 'en'）") # lang: 音频语言
):
    """
    并发处理批量音频文件的转录。

    Args:
        files: 上传的音频文件列表 (UploadFile 对象)。
        keys: 每个音频文件的名称，以逗号分隔的字符串。
        lang: 音频内容的语言，默认为自动检测 ("auto")。
        
    Returns:
        BatchApiResponse: 包含所有文件转录结果（或错误信息）的响应。
    """
    # 检查文件列表是否为空
    if not files:
        # 如果没有提供有效的音频文件，则引发HTTP 400错误
        raise HTTPException(status_code=400, detail={"error": "没有提供有效的音频文件"})
    
    # 解析keys参数，去除首尾空格，并过滤掉空字符串
    # key_list: 从逗号分隔的 keys 字符串解析出来的文件名列表
    key_list = [key.strip() for key in keys.split(',') if key.strip()]
    if len(key_list) != len(files):
        # 如果文件名数量与上传文件数量不匹配，则引发HTTP 400错误
        raise HTTPException(status_code=400, detail={"error": "音频文件数量与keys参数数量不匹配"})
    
    # tasks: 用于存储将要并发执行的异步任务的列表
    tasks = []
    # 遍历上传的文件及其对应的key，为每个文件创建一个处理任务
    for i, audio_file_item in enumerate(files): # audio_file_item: 当前遍历到的 UploadFile 对象
        # current_file_key: 当前文件对应的名称/标识符
        current_file_key = key_list[i]
        # 创建一个处理单个音频文件的异步任务，并将其添加到任务列表
        # _process_audio_file 是我们定义的辅助函数，它会异步处理单个文件
        task = _process_audio_file(audio_file_item, current_file_key, lang, model)
        tasks.append(task)
        
    # 并发执行所有创建的任务
    # asyncio.gather(*tasks, return_exceptions=True) 会等待所有任务完成。
    # return_exceptions=True 确保即使部分任务抛出异常，gather也不会立即失败，
    # 而是将异常作为对应任务的结果返回。
    # all_task_results: 包含所有任务执行结果（或异常对象）的列表
    all_task_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # results_list: 存储所有文件去除标签后的转录文本的列表
    results_list: List[str] = []
    # label_results_list: 存储所有文件原始带标签转录文本的列表
    label_results_list: List[str] = []
    # success_count: 成功处理的文件数量
    success_count: int = 0
    
    # 遍历并发任务的执行结果
    for i, task_result_item in enumerate(all_task_results): # task_result_item: 单个任务的执行结果或异常
        # current_key_for_result: 当前结果对应的文件名
        current_key_for_result = key_list[i]
        if isinstance(task_result_item, Exception):
            # 如果任务结果是一个异常对象，说明该文件处理失败
            # error_message: 格式化的错误消息字符串
            error_message = f"处理文件 {current_key_for_result} 时出错: {str(task_result_item)}"
            results_list.append(error_message) # 将错误信息添加到结果列表
            label_results_list.append(f"错误: {str(task_result_item)}") # 将更简洁的错误信息添加到标签结果列表
        else:
            # 如果任务结果不是异常，说明文件处理成功，task_result_item 是 (processed_text, raw_text) 元组
            # processed_text: 后处理过的文本
            # raw_text: 原始模型输出
            processed_text, raw_text = task_result_item
            results_list.append(processed_text) # 添加成功处理的文本
            label_results_list.append(raw_text) # 添加原始输出
            success_count += 1 # 成功计数增加
            
    # failed_count: 处理失败的文件数量
    failed_count = len(files) - success_count
    # response_message: 最终的响应消息，总结处理情况
    response_message = f"共处理 {len(files)} 个音频文件。成功: {success_count} 个，失败: {failed_count} 个。"
    
    return {
        "message": response_message,
        "results": results_list,
        "label_result": label_results_list
    }


# --- Uvicorn 服务器启动 (用于直接运行此脚本时) ---
if __name__ == "__main__":
    # 打印文档链接，方便开发者访问API文档 (Swagger UI 或 ReDoc)
    print("\n\nAPI 文档地址: http://127.0.0.1:8000/docs  或  http://127.0.0.1:8000/redoc\n")
    
    import uvicorn # 导入 uvicorn，一个ASGI服务器实现
    # 启动Uvicorn服务器来运行FastAPI应用
    # app: FastAPI应用实例
    # host="0.0.0.0": 使服务器可以从任何网络接口访问
    # port=8000: 指定服务器监听的端口号
    uvicorn.run(app, host="0.0.0.0", port=8000)