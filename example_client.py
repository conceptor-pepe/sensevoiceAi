#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SenseVoice API 客户端示例
-------------------------
此示例展示了如何使用 Python 与 SenseVoice API 进行交互，
实现音频转录、字幕生成等功能。
"""

import os
import sys
import json
import time
import argparse
import concurrent.futures
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import requests
from requests.exceptions import RequestException
from tqdm import tqdm
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('sensevoice_client')

# 尝试加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class APIError(Exception):
    """
    API错误异常类
    用于处理API返回的错误信息
    """
    message: str
    code: str
    status: int

    def __str__(self) -> str:
        return f"API错误: {self.message} (代码: {self.code}, 状态码: {self.status})"


class SenseVoiceClient:
    """
    SenseVoice API 客户端类
    封装了与SenseVoice API交互的各种方法
    """

    def __init__(self, api_url: str = None, api_key: str = None):
        """
        初始化SenseVoice客户端

        参数:
            api_url (str, 可选): API基础URL，默认从环境变量获取或使用默认值
            api_key (str, 可选): API密钥，默认从环境变量获取
        """
        # 从参数或环境变量获取API URL
        self.api_url = api_url or os.getenv('SENSEVOICE_API_URL', 'http://localhost:8000')
        
        # 删除URL末尾的斜杠
        self.api_url = self.api_url.rstrip('/')
        
        # 从参数或环境变量获取API密钥
        self.api_key = api_key or os.getenv('SENSEVOICE_API_KEY')
        
        # 初始化请求会话
        self.session = requests.Session()
        
        # 如果有API密钥，添加到请求头
        if self.api_key:
            self.session.headers.update({'Authorization': f'Bearer {self.api_key}'})
        
        logger.info(f"已初始化SenseVoice客户端，API地址: {self.api_url}")

    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None, 
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        timeout: int = 60,
        stream: bool = False
    ) -> Any:
        """
        发送HTTP请求到API并处理响应

        参数:
            method (str): HTTP方法 (GET, POST等)
            endpoint (str): API端点路径
            params (Dict, 可选): URL查询参数
            data (Dict, 可选): 表单数据或JSON数据
            files (Dict, 可选): 文件数据
            timeout (int): 请求超时时间（秒）
            stream (bool): 是否使用流式响应

        返回:
            Dict: 响应数据（JSON）

        异常:
            APIError: API返回错误
            RequestException: 请求异常
        """
        # 构建完整URL
        url = f"{self.api_url}{endpoint}"
        
        logger.debug(f"发送{method}请求到 {url}")
        
        try:
            # 发送请求
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                data=data,
                files=files,
                timeout=timeout,
                stream=stream
            )
            
            # 处理流式响应
            if stream:
                return response
            
            # 检查响应状态码
            if response.status_code >= 400:
                # 尝试解析错误响应
                try:
                    error_data = response.json()
                    message = error_data.get('error', {}).get('message', '未知错误')
                    code = error_data.get('error', {}).get('code', 'unknown_error')
                except ValueError:
                    message = response.text or f"HTTP错误: {response.status_code}"
                    code = 'http_error'
                
                raise APIError(message=message, code=code, status=response.status_code)
            
            # 返回JSON响应
            if response.headers.get('Content-Type', '').startswith('application/json'):
                return response.json()
            else:
                # 对于非JSON响应，返回原始内容
                return {'content': response.content, 'text': response.text}
            
        except RequestException as e:
            # 处理网络和请求异常
            logger.error(f"请求异常: {str(e)}")
            raise APIError(
                message=f"请求异常: {str(e)}", 
                code="request_error", 
                status=0
            )

    def health(self) -> Dict[str, Any]:
        """
        检查API健康状态

        返回:
            Dict: 健康状态信息
        """
        return self._make_request('GET', '/api/v1/health')

    def get_models(self) -> Dict[str, Any]:
        """
        获取可用的语音识别模型列表

        返回:
            Dict: 包含模型列表的响应
        """
        return self._make_request('GET', '/api/v1/models')

    def transcribe(
        self,
        file: Union[str, Path],
        model: str = "sense-voice-small",
        language: str = "auto",
        response_format: str = "json",
        timestamps: bool = False,
        word_timestamps: bool = False,
        task: str = "transcribe",
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        转录音频文件

        参数:
            file (str|Path): 音频文件路径
            model (str): 模型ID
            language (str): 音频语言代码
            response_format (str): 响应格式 (json, text, srt, vtt)
            timestamps (bool): 是否包含时间戳
            word_timestamps (bool): 是否包含单词级时间戳
            task (str): 任务类型 (transcribe, translate)
            prompt (str, 可选): 提供额外的提示或上下文

        返回:
            Dict: 转录结果
        """
        # 确保文件存在
        file_path = Path(file)
        if not file_path.exists():
            raise ValueError(f"文件不存在: {file_path}")
        
        # 准备请求数据
        data = {
            'model': model,
            'language': language,
            'response_format': response_format,
            'timestamps': 'true' if timestamps else 'false',
            'word_timestamps': 'true' if word_timestamps else 'false',
            'task': task
        }
        
        # 添加可选参数
        if prompt:
            data['prompt'] = prompt
        
        # 准备文件数据
        files = {
            'file': (file_path.name, open(file_path, 'rb'), f'audio/{file_path.suffix[1:]}')
        }
        
        try:
            # 发送转录请求
            response = self._make_request(
                'POST', 
                '/api/v1/audio/transcriptions', 
                data=data, 
                files=files,
                timeout=300  # 较长的超时时间用于大文件
            )
            
            # 关闭文件
            files['file'][1].close()
            
            return response
        except Exception as e:
            # 确保文件关闭
            files['file'][1].close()
            raise e

    def batch_transcribe(
        self,
        files: List[Union[str, Path]],
        model: str = "sense-voice-small",
        language: str = "auto",
        response_format: str = "json",
        max_workers: int = 3,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        批量转录多个音频文件

        参数:
            files (List[str|Path]): 音频文件路径列表
            model (str): 模型ID
            language (str): 音频语言代码
            response_format (str): 响应格式 (json, text)
            max_workers (int): 最大并行工作线程数
            show_progress (bool): 是否显示进度条

        返回:
            Dict: 批量转录结果
        """
        # 验证文件列表
        valid_files = []
        for file in files:
            file_path = Path(file)
            if file_path.exists():
                valid_files.append(file_path)
            else:
                logger.warning(f"跳过不存在的文件: {file_path}")
        
        if not valid_files:
            raise ValueError("没有有效的音频文件可供处理")
        
        results = {
            'results': [],
            'successful': [],
            'failed': [],
            'successful_count': 0,
            'failed_count': 0,
            'total_count': len(valid_files),
            'elapsed_time': 0
        }
        
        # 创建进度条
        pbar = tqdm(total=len(valid_files), disable=not show_progress, desc="批量转录进度")
        
        start_time = time.time()
        
        # 定义处理单个文件的函数
        def process_file(file_path):
            try:
                # 调用单文件转录方法
                result = self.transcribe(
                    file=file_path,
                    model=model,
                    language=language,
                    response_format=response_format
                )
                
                # 添加文件信息
                result['file'] = str(file_path)
                result['success'] = True
                
                # 更新进度条
                pbar.update(1)
                
                return result
            except Exception as e:
                # 更新进度条
                pbar.update(1)
                
                # 返回错误信息
                return {
                    'file': str(file_path),
                    'success': False,
                    'error': str(e)
                }
        
        # 使用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_file, file): file for file in valid_files}
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results['results'].append(result)
                
                if result['success']:
                    results['successful'].append(result)
                    results['successful_count'] += 1
                else:
                    results['failed'].append(result)
                    results['failed_count'] += 1
        
        # 关闭进度条
        pbar.close()
        
        # 计算总耗时
        results['elapsed_time'] = time.time() - start_time
        
        return results

    def save_subtitles_to_file(
        self,
        file: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        format: str = "srt",
        model: str = "sense-voice-small",
        language: str = "auto"
    ) -> str:
        """
        生成字幕并保存到文件

        参数:
            file (str|Path): 音频/视频文件路径
            output_path (str|Path, 可选): 输出字幕文件路径，如果为None则根据输入文件生成
            format (str): 字幕格式 (srt, vtt)
            model (str): 模型ID
            language (str): 音频语言代码

        返回:
            str: 保存的字幕文件路径
        """
        # 检查格式有效性
        if format not in ['srt', 'vtt']:
            raise ValueError(f"不支持的字幕格式: {format}，请使用 'srt' 或 'vtt'")
        
        # 准备输出路径
        file_path = Path(file)
        if not output_path:
            output_path = file_path.with_suffix(f'.{format}')
        else:
            output_path = Path(output_path)
        
        # 调用转录方法，指定字幕格式
        result = self.transcribe(
            file=file_path,
            model=model,
            language=language,
            response_format=format,
            timestamps=True  # 字幕必须包含时间戳
        )
        
        # 对于JSON响应，内容应该在text字段
        if isinstance(result, dict) and 'text' in result:
            subtitle_content = result['text']
        # 对于非JSON响应，内容应该在text字段
        elif isinstance(result, dict) and 'text' in result:
            subtitle_content = result['text']
        else:
            # 处理意外的响应格式
            raise ValueError(f"无法从API响应中获取字幕内容: {result}")
        
        # 保存字幕到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(subtitle_content)
        
        logger.info(f"字幕已保存到: {output_path}")
        
        return str(output_path)

    def get_performance_stats(self, summary: bool = False) -> Dict[str, Any]:
        """
        获取API性能统计数据

        参数:
            summary (bool): 是否只返回摘要信息

        返回:
            Dict: 性能统计数据
        """
        endpoint = '/api/v1/stats/performance'
        if summary:
            endpoint += '?summary=true'
            
        return self._make_request('GET', endpoint)

    def get_system_metrics(self, summary: bool = False) -> Dict[str, Any]:
        """
        获取系统指标

        参数:
            summary (bool): 是否只返回摘要信息

        返回:
            Dict: 系统指标数据
        """
        endpoint = '/api/v1/stats/system'
        if summary:
            endpoint += '?summary=true'
            
        return self._make_request('GET', endpoint)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        返回:
            Dict: 缓存统计数据
        """
        return self._make_request('GET', '/api/v1/cache/stats')

    def clear_cache(self) -> Dict[str, Any]:
        """
        清除API缓存

        返回:
            Dict: 操作结果
        """
        return self._make_request('POST', '/api/v1/cache/clear')

    def save_result_to_file(self, result: Dict[str, Any], output_path: Union[str, Path]) -> str:
        """
        将结果保存到文件

        参数:
            result (Dict): 要保存的结果
            output_path (str|Path): 输出文件路径

        返回:
            str: 保存的文件路径
        """
        output_path = Path(output_path)
        
        # 确保目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到: {output_path}")
        
        return str(output_path)


def display_result(result: Dict[str, Any], format: str = 'pretty') -> None:
    """
    显示结果

    参数:
        result (Dict): 要显示的结果
        format (str): 显示格式 (pretty, raw)
    """
    if format == 'raw':
        print(json.dumps(result, ensure_ascii=False))
    else:
        # 尝试漂亮地打印结果
        try:
            if 'text' in result:
                print("\n=== 转录文本 ===")
                print(result['text'])
                print("\n=== 其他信息 ===")
                for key, value in result.items():
                    if key != 'text':
                        print(f"{key}: {value}")
            elif 'results' in result and isinstance(result['results'], list):
                print("\n=== 批处理结果摘要 ===")
                print(f"总文件数: {result.get('total_count', len(result['results']))}")
                print(f"成功: {result.get('successful_count', 0)}")
                print(f"失败: {result.get('failed_count', 0)}")
                print(f"总耗时: {result.get('elapsed_time', 0):.2f} 秒")
                
                if result.get('failed', []):
                    print("\n=== 失败的文件 ===")
                    for item in result.get('failed', []):
                        print(f"{item.get('file')}: {item.get('error')}")
            else:
                # 通用JSON打印
                print(json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as e:
            # 如果漂亮打印失败，回退到原始打印
            logger.debug(f"漂亮打印失败: {e}")
            print(json.dumps(result, ensure_ascii=False))


def main():
    """主函数，处理命令行参数和调用相应的功能"""
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='SenseVoice API 客户端工具')
    
    # 添加子命令解析器
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 全局选项
    parser.add_argument('--api-url', help='API 基础 URL')
    parser.add_argument('--api-key', help='API 密钥')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    # 健康检查子命令
    health_parser = subparsers.add_parser('health', help='检查 API 健康状态')
    
    # 模型列表子命令
    models_parser = subparsers.add_parser('models', help='获取可用模型列表')
    
    # 转录子命令
    transcribe_parser = subparsers.add_parser('transcribe', help='转录音频文件')
    transcribe_parser.add_argument('file', help='音频文件路径')
    transcribe_parser.add_argument('--model', default='sense-voice-small', help='使用的模型 ID')
    transcribe_parser.add_argument('--language', default='auto', help='音频语言 (auto, zh, en, yue, ja, ko)')
    transcribe_parser.add_argument('--format', default='json', help='输出格式 (json, text, srt, vtt)')
    transcribe_parser.add_argument('--output', '-o', help='输出文件路径')
    transcribe_parser.add_argument('--timestamps', action='store_true', help='包含时间戳')
    transcribe_parser.add_argument('--word-timestamps', action='store_true', help='包含单词级时间戳')
    transcribe_parser.add_argument('--task', default='transcribe', help='任务类型 (transcribe, translate)')
    transcribe_parser.add_argument('--prompt', help='提供提示以引导转录')
    
    # 批量转录子命令
    batch_parser = subparsers.add_parser('batch', help='批量转录多个音频文件')
    batch_parser.add_argument('files', nargs='+', help='音频文件路径列表')
    batch_parser.add_argument('--model', default='sense-voice-small', help='使用的模型 ID')
    batch_parser.add_argument('--language', default='auto', help='音频语言')
    batch_parser.add_argument('--format', default='json', help='输出格式 (json, text)')
    batch_parser.add_argument('--workers', type=int, default=3, help='并行工作线程数')
    batch_parser.add_argument('--output', '-o', help='输出 JSON 文件路径')
    
    # 字幕生成子命令
    subtitles_parser = subparsers.add_parser('subtitles', help='生成字幕文件')
    subtitles_parser.add_argument('file', help='视频或音频文件路径')
    subtitles_parser.add_argument('--format', default='srt', choices=['srt', 'vtt'], help='字幕格式 (srt, vtt)')
    subtitles_parser.add_argument('--model', default='sense-voice-small', help='使用的模型 ID')
    subtitles_parser.add_argument('--language', default='auto', help='音频语言')
    subtitles_parser.add_argument('--output', '-o', help='输出字幕文件路径')
    
    # 性能统计子命令
    performance_parser = subparsers.add_parser('performance', help='获取性能统计')
    performance_parser.add_argument('--summary', action='store_true', help='只返回摘要信息')
    performance_parser.add_argument('--output', '-o', help='输出JSON文件路径')
    
    # 系统指标子命令
    system_parser = subparsers.add_parser('system', help='获取系统指标')
    system_parser.add_argument('--summary', action='store_true', help='只返回摘要信息')
    system_parser.add_argument('--output', '-o', help='输出JSON文件路径')
    
    # 缓存操作子命令
    cache_parser = subparsers.add_parser('cache', help='缓存操作')
    cache_parser.add_argument('--clear', action='store_true', help='清除缓存')
    cache_parser.add_argument('--output', '-o', help='输出JSON文件路径')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 设置日志级别
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("调试模式已启用")
    
    # 创建客户端实例
    client = SenseVoiceClient(api_url=args.api_url, api_key=args.api_key)
    
    try:
        # 根据子命令执行相应的功能
        if args.command == 'health':
            # 检查API健康状态
            result = client.health()
            display_result(result)
            
        elif args.command == 'models':
            # 获取可用模型列表
            result = client.get_models()
            display_result(result)
            
        elif args.command == 'transcribe':
            # 转录单个音频文件
            result = client.transcribe(
                file=args.file,
                model=args.model,
                language=args.language,
                response_format=args.format,
                timestamps=args.timestamps,
                word_timestamps=args.word_timestamps,
                task=args.task,
                prompt=args.prompt
            )
            
            # 显示结果
            display_result(result)
            
            # 如果指定了输出文件，保存结果
            if args.output:
                client.save_result_to_file(result, args.output)
                
        elif args.command == 'batch':
            # 批量转录多个音频文件
            result = client.batch_transcribe(
                files=args.files,
                model=args.model,
                language=args.language,
                response_format=args.format,
                max_workers=args.workers,
                show_progress=True
            )
            
            # 显示结果摘要
            display_result(result)
            
            # 如果指定了输出文件，保存结果
            if args.output:
                client.save_result_to_file(result, args.output)
                
        elif args.command == 'subtitles':
            # 生成字幕文件
            output_path = client.save_subtitles_to_file(
                file=args.file,
                output_path=args.output,
                format=args.format,
                model=args.model,
                language=args.language
            )
            print(f"字幕已保存到: {output_path}")
            
        elif args.command == 'performance':
            # 获取性能统计
            result = client.get_performance_stats(summary=args.summary)
            display_result(result)
            
            # 如果指定了输出文件，保存结果
            if args.output:
                client.save_result_to_file(result, args.output)
                
        elif args.command == 'system':
            # 获取系统指标
            result = client.get_system_metrics(summary=args.summary)
            display_result(result)
            
            # 如果指定了输出文件，保存结果
            if args.output:
                client.save_result_to_file(result, args.output)
                
        elif args.command == 'cache':
            # 缓存操作
            if args.clear:
                # 清除缓存
                result = client.clear_cache()
                print("缓存已清除")
            else:
                # 获取缓存统计
                result = client.get_cache_stats()
            
            display_result(result)
            
            # 如果指定了输出文件，保存结果
            if args.output:
                client.save_result_to_file(result, args.output)
                
        else:
            # 如果没有提供有效的子命令，显示帮助
            parser.print_help()
            
    except APIError as e:
        # 处理API错误
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        # 处理其他异常
        logger.error(f"错误: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 