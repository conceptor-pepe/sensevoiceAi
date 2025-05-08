import librosa
import numpy as np
# import aiohttp # aiohttp 在当前代码片段中未被使用，如果后续需要可以取消注释
from fastapi import FastAPI, Form, UploadFile, HTTPException
from pydantic import HttpUrl, ValidationError, BaseModel, Field
from typing import List, Union, Any, Dict
from funasr_onnx import SenseVoiceSmall
from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess
from io import BytesIO
import asyncio # 导入 asyncio 库，用于异步编程
import functools # 导入 functools 库，用于 functools.partial
import concurrent.futures # 导入 concurrent.futures 模块，用于创建线程池
import time # 导入 time 模块，用于时间统计
import os # 导入 os 模块，用于文件和目录操作
from datetime import datetime # 导入 datetime 模块，用于日期时间格式化

# 导入配置和日志模块
import config
from logger import logger

# --- Pydantic 模型定义 ---
class ApiResponse(BaseModel):
    """单文件API响应模型 (当前未使用，但保留定义以备将来扩展)"""
    message: str = Field(..., description="状态消息，指示操作的成功。") # message: 操作状态消息
    results: str = Field(..., description="去除标签的输出。") # results: 去除时间戳和标点等标签后的文本结果
    label_result: str = Field(..., description="默认输出。") # label_result: 原始的、带标签的文本结果
    processing_time: float = Field(0.0, description="处理时间（秒）") # processing_time: 处理音频文件所需的时间（秒）

class BatchApiResponse(BaseModel):
    """批量API响应模型"""
    message: str = Field(..., description="操作状态消息，可能包含成功和失败的计数。") # message: 操作状态消息，例如 "成功处理 X 个文件中的 Y 个，失败 Z 个"
    results: List[str] = Field(..., description="去除标签的输出结果列表，每个元素对应一个文件。") # results: 存储所有文件去除标签后的转录文本列表
    label_result: List[str] = Field(..., description="原始输出结果列表，每个元素对应一个文件。") # label_result: 存储所有文件原始的、带标签的转录文本列表
    processing_time: float = Field(0.0, description="处理总时间（秒）") # processing_time: 处理所有音频文件所需的总时间（秒）
    file_times: List[float] = Field([], description="每个文件的处理时间（秒）") # file_times: 每个文件的处理时间列表

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

# 记录模型加载开始
logger.info(f"Begin to load model: {config.MODEL_DIR}, batch_size: {config.BATCH_SIZE}, quantize: {config.QUANTIZE}")
model_load_start_time = time.time()

# model: SenseVoiceSmall 模型的全局实例
model = SenseVoiceSmall(
    config.MODEL_DIR,
    quantize=config.QUANTIZE,
    device_id=config.DEVICE_ID,
    batch_size=config.BATCH_SIZE
)

# 记录模型加载结束和耗时
model_load_time = time.time() - model_load_start_time
logger.info(f"Model loaded, time cost: {model_load_time:.4f} seconds")

# --- 全局线程池执行器 ---
# model_executor: 初始化一个全局的线程池执行器，专门用于处理模型推理等阻塞型CPU/GPU密集任务
# max_workers=MODEL_WORKERS: 限制了同时执行模型推理的线程数量，对于GPU任务，通常设为1
model_executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.MODEL_WORKERS)
logger.info(f"Initialize thread pool executor, thread number: {config.MODEL_WORKERS}")

# --- FastAPI 应用实例 ---
app = FastAPI() # app: FastAPI 应用的主实例

@app.on_event("startup")
async def startup_event():
    """应用程序启动时调用的事件处理器"""
    logger.info("SenseVoice API 服务启动")

@app.on_event("shutdown")
async def app_shutdown():
    """
    应用程序关闭时调用的事件处理器。
    负责优雅地关闭全局模型推理线程池。
    """
    global model_executor # 引用全局执行器实例
    if model_executor:
        logger.info("Closing model inference thread pool...") # 记录日志：正在关闭线程池
        # model_executor.shutdown(wait=True) 会等待所有已提交的任务完成后再关闭线程池。
        # 这对于确保所有正在进行的推理任务都能完成非常重要。
        model_executor.shutdown(wait=True)
        logger.info("Model inference thread pool closed.") # 记录日志：线程池已关闭
    logger.info("SenseVoice API service closed")

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
    # 记录开始处理时间
    file_start_time = time.time()
    file_size = 0
    
    try:
        # audio_content: 从 UploadFile 异步读取的音频文件字节内容
        audio_content: bytes = await audio_file.read()
        file_size = len(audio_content)
        # 记录文件信息
        logger.info(f"开始处理文件: {key}, 大小: {file_size/1024:.2f} KB, 语言: {lang}")
        
        # audio_fp: 将字节内容包装成的 BytesIO 对象，方便模型读取和处理
        audio_fp = BytesIO(audio_content)
        
        # 获取当前 asyncio 事件循环实例
        loop = asyncio.get_event_loop()

        # model_func_partial: 创建一个偏函数，预设 model_instance 的 language 和 use_itn 参数。
        model_func_partial = functools.partial(model_instance, language=lang, use_itn=True)
        
        # 记录模型推理开始
        inference_start_time = time.time()
        
        # res: 模型对音频文件进行语音识别的原始结果。
        # 使用 loop.run_in_executor 将阻塞的 model_func_partial(audio_fp) 调用
        # 放入我们定义的 model_executor 线程池中执行。
        res = await loop.run_in_executor(
            model_executor,     # executor: 使用全局定义的、固定大小的线程池
            model_func_partial, # func: 要在线程池中执行的函数 (已绑定参数的偏函数)
            audio_fp            # *args: 传递给 func 的位置参数 (这里是音频数据)
        )
        
        # 记录模型推理耗时
        inference_time = time.time() - inference_start_time
        
        # processed_text: 对原始识别结果 res[0] 进行后处理（例如，去除标签、规范化文本格式）后的文本。
        processed_text: str = rich_transcription_postprocess(res[0])
        # raw_text: 未经 rich_transcription_postprocess 处理的原始模型输出。
        raw_text = res[0] # 通常 res[0] 直接就是原始的带标签文本或包含更丰富信息的结构
        
        # 计算处理总耗时
        file_process_time = time.time() - file_start_time
        logger.info(f"File {key} processed, model inference: {inference_time:.4f} seconds, total time cost: {file_process_time:.4f} seconds")

        return processed_text, raw_text, file_process_time # 返回成功处理的结果和处理时间
    except Exception as e:
        # 记录错误
        error_message = f"Error occurred while processing file {key}: {str(e)}"
        logger.error(error_message)
        # 计算处理总耗时（即使失败）
        file_process_time = time.time() - file_start_time
        logger.info(f"File {key} processing failed, total time cost: {file_process_time:.4f} seconds")
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
    # 记录请求开始时间
    request_start_time = time.time()
    
    # 获取客户端IP地址 (简化处理)
    client_ip = "未知IP"
    
    # 记录访问日志
    logger.info(f"Request: client IP={client_ip}, file number={len(files)}, language={lang}")
    
    # 验证请求参数
    key_list = _validate_request_params(files, keys, client_ip)
    
    # 创建并执行异步任务
    all_task_results = await _execute_transcription_tasks(files, key_list, lang, model)
    
    # 处理任务结果
    results = _process_task_results(all_task_results, key_list)
    
    # 计算总处理时间
    total_process_time = time.time() - request_start_time
    
    # 生成响应消息
    response_message = _generate_response_message(len(files), results["success_count"], 
                                                results["failed_count"], total_process_time)
    
    # 记录性能统计信息
    _log_performance_stats(results["file_times_list"], client_ip, total_process_time, 
                          results["success_count"], results["failed_count"])
    
    return {
        "message": response_message,
        "results": results["results_list"],
        "label_result": results["label_results_list"],
        "processing_time": total_process_time,
        "file_times": results["file_times_list"]
    }

def _validate_request_params(files: List[UploadFile], keys: str, client_ip: str) -> List[str]:
    """
    验证请求参数的有效性。
    
    Args:
        files: 上传的音频文件列表
        keys: 文件名称字符串，以逗号分隔
        client_ip: 客户端IP地址
        
    Returns:
        List[str]: 解析后的文件名列表
        
    Raises:
        HTTPException: 当参数无效时抛出
    """
    # 检查文件列表是否为空
    if not files:
        # 如果没有提供有效的音频文件，则引发HTTP 400错误
        logger.warning(f"Request error: client IP={client_ip}, no valid audio files provided")
        raise HTTPException(status_code=400, detail={"error": "没有提供有效的音频文件"})
    
    # 解析keys参数，去除首尾空格，并过滤掉空字符串
    key_list = [key.strip() for key in keys.split(',') if key.strip()]
    if len(key_list) != len(files):
        # 如果文件名数量与上传文件数量不匹配，则引发HTTP 400错误
        logger.warning(f"Request error: client IP={client_ip}, audio file number({len(files)}) does not match the number of keys parameters({len(key_list)})")
        raise HTTPException(status_code=400, detail={"error": "Audio file number does not match the number of keys parameters"})
    
    return key_list

async def _execute_transcription_tasks(files: List[UploadFile], key_list: List[str], 
                                     lang: str, model_instance) -> List[Any]:
    """
    创建并执行音频转录的异步任务。
    
    Args:
        files: 上传的音频文件列表
        key_list: 文件名列表
        lang: 音频语言
        model_instance: 转录模型实例
        
    Returns:
        List[Any]: 所有任务的执行结果
    """
    # tasks: 用于存储将要并发执行的异步任务的列表
    tasks = []
    # 遍历上传的文件及其对应的key，为每个文件创建一个处理任务
    for i, audio_file_item in enumerate(files): # audio_file_item: 当前遍历到的 UploadFile 对象
        # current_file_key: 当前文件对应的名称/标识符
        current_file_key = key_list[i]
        # 创建一个处理单个音频文件的异步任务，并将其添加到任务列表
        task = _process_audio_file(audio_file_item, current_file_key, lang, model_instance)
        tasks.append(task)
    
    logger.info(f"Create {len(tasks)} async task")
    
    # 并发执行所有创建的任务
    return await asyncio.gather(*tasks, return_exceptions=True)

def _process_task_results(all_task_results: List[Any], key_list: List[str]) -> Dict[str, Any]:
    """
    处理任务执行结果，整理成响应格式。
    
    Args:
        all_task_results: 任务执行结果列表
        key_list: 文件名列表
        
    Returns:
        Dict: 包含处理结果的字典
    """
    # results_list: 存储所有文件去除标签后的转录文本的列表
    results_list: List[str] = []
    # label_results_list: 存储所有文件原始带标签转录文本的列表
    label_results_list: List[str] = []
    # file_times_list: 存储每个文件处理时间的列表
    file_times_list: List[float] = []
    # success_count: 成功处理的文件数量
    success_count: int = 0
    
    # 遍历并发任务的执行结果
    for i, task_result_item in enumerate(all_task_results): # task_result_item: 单个任务的执行结果或异常
        # current_key_for_result: 当前结果对应的文件名
        current_key_for_result = key_list[i]
        if isinstance(task_result_item, Exception):
            # 如果任务结果是一个异常对象，说明该文件处理失败
            # error_message: 格式化的错误消息字符串
            error_message = f"File processing error: file={current_key_for_result}, error={str(task_result_item)}"
            results_list.append(error_message) # 将错误信息添加到结果列表
            label_results_list.append(f"错误: {str(task_result_item)}") # 将更简洁的错误信息添加到标签结果列表
            file_times_list.append(0.0) # 对于失败的处理添加0作为处理时间
            logger.error(f"File processing error: file={current_key_for_result}, error={str(task_result_item)}")
        else:
            # 如果任务结果不是异常，说明文件处理成功，task_result_item 是 (processed_text, raw_text, process_time) 元组
            processed_text, raw_text, process_time = task_result_item
            results_list.append(processed_text) # 添加成功处理的文本
            label_results_list.append(raw_text) # 添加原始输出
            file_times_list.append(process_time) # 添加处理时间
            success_count += 1 # 成功计数增加
    
    # failed_count: 处理失败的文件数量
    failed_count = len(all_task_results) - success_count
    
    return {
        "results_list": results_list,
        "label_results_list": label_results_list,
        "file_times_list": file_times_list,
        "success_count": success_count,
        "failed_count": failed_count
    }

def _generate_response_message(total_files: int, success_count: int, 
                             failed_count: int, total_time: float) -> str:
    """
    生成响应消息，总结处理情况。
    
    Args:
        total_files: 总文件数
        success_count: 成功处理的文件数
        failed_count: 处理失败的文件数
        total_time: 总处理时间
        
    Returns:
        str: 格式化的响应消息
    """
    response_message = f"Total {total_files} audio files processed. Success: {success_count} files, Failed: {failed_count} files. Total time cost: {total_time:.4f} seconds."
    logger.info(response_message)
    return response_message

def _log_performance_stats(file_times: List[float], client_ip: str, 
                         total_time: float, success_count: int, failed_count: int) -> None:
    """
    记录性能统计信息。
    
    Args:
        file_times: 文件处理时间列表
        client_ip: 客户端IP
        total_time: 总处理时间
        success_count: 成功处理的文件数
        failed_count: 处理失败的文件数
    """
    if file_times:
        avg_time = sum(file_times) / len(file_times)
        max_time = max(file_times)
        min_time = min(filter(lambda x: x > 0, file_times or [0]))
        logger.info(f"File handle cost - avg: {avg_time:.4f}second, max: {max_time:.4f}, min: {min_time:.4f}")
    
    # 记录访问日志 - 请求完成
    logger.info(f"handle: Client IP={client_ip}, Total cost={total_time:.4f}second, success={success_count}, failed={failed_count}")


# --- Uvicorn 服务器启动 (用于直接运行此脚本时) ---
if __name__ == "__main__":
    # 打印文档链接
    logger.info("\n\nAPI 文档地址: http://127.0.0.1:8000/docs  或  http://127.0.0.1:8000/redoc\n")
    
    import uvicorn # 导入 uvicorn，一个ASGI服务器实现
    # 启动Uvicorn服务器来运行FastAPI应用
    uvicorn.run(app, host=config.HOST, port=config.PORT)