# SenseVoice API 流式识别功能说明

SenseVoice API现在支持三种流式语音识别方式：

1. HTTP流式响应 - 适用于离线音频文件的流式处理
2. WebSocket实时交互 - 适用于实时语音流（如麦克风输入）
3. 兼容GitHub SenseVoice API的流式接口

本文档详细说明了如何使用这些流式功能。

## 1. HTTP流式识别接口

### 基本流式接口 `/recognize/stream`

这个接口适用于已有的音频文件，服务端会将文件分块处理并以流的形式返回识别结果。

**请求方式**：

1. **上传音频文件**:
   ```bash
   curl -X POST "http://localhost:8000/recognize/stream" \
     -F "audio_file=@/path/to/your/audio.wav" \
     -F "language=auto" \
     -F "use_itn=true" \
     -F "chunk_size_sec=3"
   ```

2. **使用Base64编码**:
   ```bash
   curl -X POST "http://localhost:8000/recognize/stream" \
     -F 'request_data={"audio_base64":"BASE64_ENCODED_AUDIO_DATA", "language":"auto", "use_itn":true, "chunk_size_sec":3}'
   ```

### 兼容接口 `/api/v1/asr/stream`

```bash
curl -X POST "http://localhost:8000/api/v1/asr/stream" \
  -F "files=@/path/to/your/audio.wav" \
  -F "keys=audio1" \
  -F "lang=auto" \
  -F "use_itn=true" \
  -F "chunk_size_sec=3"
```

**参数说明**：

- `audio_file` 或 `files`: 需要识别的音频文件
- `language` 或 `lang`: 音频语言，可选值为`auto`、`zh`、`en`等，默认为`auto`
- `use_itn`: 是否使用反向文本归一化，默认为`true`
- `chunk_size_sec`: 分块大小（秒），默认为3秒

**响应格式**：

返回的是新行分隔的JSON（NDJSON）格式，每行包含一个JSON对象，表示一个处理块的结果：

```json
{"success":true,"message":"部分识别结果","text":"你好","accumulated_text":"你好","language":"zh","emotion":"NEUTRAL","event":"Speech","is_final":false,"chunk_id":1,"time_cost":0.345}
{"success":true,"message":"部分识别结果","text":"世界","accumulated_text":"你好 世界","language":"zh","emotion":"NEUTRAL","event":"Speech","is_final":false,"chunk_id":2,"time_cost":0.326}
{"success":true,"message":"识别完成","text":"你好 世界","accumulated_text":"你好 世界","language":"zh","emotion":"NEUTRAL","event":"Speech","is_final":true,"chunk_id":2,"time_cost":0.721,"detail_time":{"流式处理开始":0.001,"处理第1块":0.345,"处理第2块":0.326,"流式处理完成":0.049}}
```

## 2. WebSocket流式识别接口

WebSocket接口提供真正的双向实时通信能力，适合处理实时音频流，如麦克风输入。

### 标准WebSocket接口 `/ws/recognize`

**连接建立**：

```javascript
const socket = new WebSocket('ws://localhost:8000/ws/recognize');
```

**认证和初始化**：

连接建立后，首先发送配置信息：

```javascript
const config = {
    language: "auto",
    use_itn: true
};
socket.send(JSON.stringify(config));
```

**发送音频数据**：

连接成功后，可以开始分块发送音频数据。每个音频块应该是二进制数据（ArrayBuffer或Blob）：

```javascript
// 发送音频块
socket.send(audioChunk);

// 发送空数据表示音频结束
socket.send(new ArrayBuffer(0));
```

**接收识别结果**：

服务器会实时返回识别结果：

```javascript
socket.onmessage = function(event) {
    const result = JSON.parse(event.data);
    
    if (result.is_final) {
        // 最终结果处理
        console.log("最终识别结果:", result.text);
    } else {
        // 部分结果处理
        console.log("当前识别:", result.text);
    }
};
```

### 兼容接口 `/api/v1/ws/asr`

兼容GitHub SenseVoice API的WebSocket接口，流程相似但响应格式略有不同：

```javascript
const socket = new WebSocket('ws://localhost:8000/api/v1/ws/asr');

// 发送配置
const config = {
    lang: "auto",
    use_itn: false,
    key: "my_audio"
};
socket.send(JSON.stringify(config));

// 接收结果
socket.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.result && data.result.length > 0) {
        console.log("识别结果:", data.result[0].text);
    }
};
```

## 3. 示例代码

我们提供了两个示例客户端来演示如何使用WebSocket流式接口：

### Python客户端示例

文件：`websocket_client_example.py`

使用方法：
```bash
python websocket_client_example.py test.wav zh true
```

### Web客户端示例

文件：`websocket_client_example.html`

直接在浏览器中打开此HTML文件，可以通过麦克风录音或上传音频文件进行流式识别演示。

## 4. 性能和调优建议

1. **音频块大小**：
   - 对于实时性要求高的场景，推荐使用较小的块大小（如0.2-0.5秒）
   - 对于准确性要求高的场景，可以使用较大的块大小（如1-3秒）

2. **采样率和格式**：
   - 最佳音频格式为16kHz采样率、16位深度、单声道WAV格式
   - 发送前最好进行音频预处理，如降噪、重采样等

3. **并发连接数**：
   - WebSocket接口对服务器资源消耗较大，请控制并发连接数

4. **网络延迟**：
   - 如果网络条件不佳，建议增加音频缓冲大小，减少发送频率

## 5. 注意事项

1. WebSocket流式识别适合音频实时流，如麦克风输入
2. HTTP流式识别适合离线音频文件的处理
3. 所有临时文件会在处理完成后自动清理
4. 当前实现仅支持WAV格式的音频

## 6. 错误处理

常见错误及解决方案：

1. **连接拒绝**：检查服务器地址和端口是否正确
2. **认证失败**：检查配置信息格式
3. **音频格式错误**：确保使用16kHz采样率的WAV格式
4. **识别失败**：查看服务器日志，可能是模型加载问题

如需更多帮助，请参考完整的API文档或联系支持团队。 