# -*- coding: utf-8 -*-
import os, re, json, time, uuid, base64, ffmpeg, dotenv, requests
from pathlib import Path
from datetime import datetime
from IPython import embed

from apis.logger import SmartLogger

dotenv.load_dotenv()


class VideoParser(object):
    
    def __init__(self, 
                 api_key='', 
                 api_base='', 
                 model_name='', 
                 log_file='./tmp/video_parser.log'):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.base_url = f"{self.api_base}/chat/completions"
        self.logger = SmartLogger('VideoParser', log_file)
        self.timeout = 300.0
    
    def compress_video(self, inp_file, tar_h=720, tar_fps=15):
        ## step1. 归置目标文件
        if not os.path.exists(inp_file):
            raise FileNotFoundError(f"输入文件不存在: {inp_file}")
        file_obj = Path(inp_file)
        outp_dir = os.path.join(file_obj.parent, "compressed_videos")
        os.makedirs(outp_dir, exist_ok=True)
        out_file = os.path.join(outp_dir, file_obj.name)
        if not os.path.exists(out_file):
            try:
                ## step2. 获取视频原始信息
                probe = ffmpeg.probe(inp_file)
                video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
                if not video_stream:
                    raise AttributeError("未从%s中找到视频流 ..." % inp_file)
                src_w, src_h = [video_stream[k] for k in ['width', 'height']]
                r_frame_rate = video_stream.get('r_frame_rate', '30/1')
                num, den = map(int, r_frame_rate.split('/'))
                orig_fps = num / den if den != 0 else 0
                file_size = os.path.getsize(inp_file)
                self.logger('debug', f"原始信息: 大小 {file_size}，分辨率 {src_w}x{src_h}, 帧率 {orig_fps:.2f} fps")

                ## step3. 分辨率
                stream = ffmpeg.input(inp_file)
                if src_h > tar_h:
                    video = stream.video.filter('scale', -2, tar_h)  # 宽度自动且为2的倍数，高度tar_h
                    self.logger('debug', f"🔔 将触发分辨率压缩: 缩放至高度 {tar_h}")
                else:
                    self.logger('debug', "🔔 分辨率未超标，保持原始分辨率")
                    video = stream.video

                ## step4. 帧率
                if orig_fps > tar_fps:
                    self.logger('debug', f"🔔 将触发降帧: {orig_fps:.2f} -> {tar_fps}")
                    video = video.filter('fps', fps=tar_fps, round='up')
                else:
                    self.logger('debug', "🔔 帧率已较低，保持原始帧率")

                ## step5. 执行命令
                out = ffmpeg.output(
                    video, 
                    stream.audio,
                    out_file, 
                    vcodec='libx264', 
                    acodec='copy', 
                    crf=23, 
                    preset='medium' 
                )
                out.run(overwrite_output=True, capture_stdout=True, capture_stderr=True)

            except Exception as e:
                self.logger('error', '❌ %s' % str(e))
            
            if not os.path.exists(out_file):
                raise IOError('视频压缩失败，未发现压缩输出文件 %s' % out_file)
        
        self.logger('info', '🟢 视频数据压缩完毕, 文件保存在源视频同级目录下的compressed_videos')
        return out_file
    
    def get_video_info(self, video_path):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        file_path = Path(video_path)
        file_size = os.path.getsize(video_path)
        extension = file_path.suffix.lower()

        mime_map = {
            ".mp4": "video/mp4",
            ".avi": "video/x-msvideo",
            ".mov": "video/quicktime",
            ".webm": "video/webm",
            ".mkv": "video/x-matroska",
            ".flv": "video/x-flv",
        }
        mime_type = mime_map.get(extension, "video/mp4")
        size_mb = file_size / (1024.0 * 1024)
        video_info = {
            "path": str(file_path),
            "name": file_path.name,
            "size_bytes": file_size,
            "size_mb": size_mb,
            "mime_type": mime_type,
            "extension": extension,
            "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        }
        self.logger('info', f"🟢 元数据: {file_path.name} ({size_mb:.2f} MB, {mime_type})")
        return video_info
    
    def encode_video_to_base64(self, video_info):
        with open(video_info['path'], "rb") as f:
            data = f.read()
        f.close()
        video_base64 = base64.b64encode(data).decode("utf-8")
        video_url = "data:%s;base64,%s" % (video_info['mime_type'], video_base64)
        video_info['video_url'] = video_url
        bs_volume = len(video_url) / (1024.0 * 1024.0)
        self.logger('info', '🟢 已完成视频流转baseb4操作, 体积 %.4f MB' % bs_volume)
    
    def generate_payload(self, 
                         video_info, 
                         prompt,
                         temperature=1.0,
                         reasoning_effort="high",
                         include_thoughts=True,
                         max_tokens=None):
        '''
        reasoning_effort : {low, minimal, medium, high, disable, none}
        '''
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url":{
                                "url": video_info['video_url'], 
                                "mime_type": video_info["mime_type"]
                            }
                        }
                    ]
                }
            ],
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
            "thinkingConfig": {"includeThoughts": include_thoughts},
            "stream": True
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        self.logger('info', '🟢 payload 装填完毕 ...')
        return payload
    
    def request_api(self, payload):
        start_time = time.time()
        response = {"success": True, "error": "", 'response_time':self.timeout}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        try:
            api_response = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout, stream=True)
            response['success'] = True if api_response.status_code == 200 else False
            api_data = {"id": None, "model": None, "content": "", "finish_reason": None}
            for line in api_response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                text = line.decode('utf-8') if isinstance(line, bytes) else line
                if text.startswith('data:'):
                    text = text[len('data:'):].strip()
                    if text == '[DONE]':
                        break
                    if not text:
                        continue
                    try:
                        obj = json.loads(text)
                        ## 提取元数据
                        if not api_data["id"] and "id" in obj:
                            api_data["id"] = obj["id"]
                            api_data["model"] = obj.get("model")
                        ## 提取内容（假设为OpenAI兼容格式）
                        if 'choices' in obj and obj['choices']:
                            choice = obj['choices'][0]
                            ## 提取增量内容
                            if 'delta' in choice:
                                chunk = choice['delta'].get('content', '') or choice['delta'].get('text', '')
                            elif 'text' in choice:
                                chunk = choice.get('text', '')
                            else:
                                chunk = ''
                            if chunk:
                                api_data["content"] += chunk
                            ## 完成原因
                            if obj['choices'][0].get('finish_reason'):
                                api_data["finish_reason"] = obj['choices'][0]['finish_reason']
                        # api_data["chunks"].append(obj)
                    except json.JSONDecodeError as e:
                        self.logger('error', f"🔴 解析chunk失败: {e}")
                        continue
            response.update(api_data)
        except requests.exceptions.Timeout:
            response['success'] = False
            response['error'] = f"请求超时（{self.timeout}秒）"
        response['response_time'] = float(time.time() - start_time) / 60.0
        self.logger('info', '✅ api 请求完成, 耗时 %.4f mins ...' % response['response_time'])
        return response

    def parse_content(self, content):
        """解析可能带有markdown标记的JSON内容"""
        if isinstance(content, str):
            # 尝试多种模式匹配
            patterns = [
                r'```json\s*([\s\S]*?)\s*```',  # ```json ... ```
                r'```\s*([\s\S]*?)\s*```',      # ``` ... ```
                r'(\[[\s\S]*\])',                 # 数组格式
                r'(\{[\s\S]*\})'                   # 对象格式
            ]
            
            for pattern in patterns:
                match = re.search(pattern, content)
                if match:
                    json_str = match.group(1).strip()
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        continue
            
            # 如果都没有匹配，尝试直接解析
            try:
                return json.loads(content.strip())
            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {e}")
                return None
        else:
            # 如果content已经是字典/列表，直接返回
            return content

    def parse_response(self, response):
        response['parse_info'] = {}
        if response['success']:
            json_data = {}
            if ('choices' in response) and (len(response['choices']) > 0):
                choice = response["choices"][0]
                if ('message' in choice) and ('content' in choice['message']):
                    # OpenAI兼容格式
                    content = choice["message"]["content"]
                    if isinstance(content, list):
                        # 如果content是列表，提取文本部分
                        text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                        return "\n".join(text_parts)
                    elif isinstance(content, str):
                        json_data = json.loads(content[8:-4])  ## '```json\n{...}\n```'
            elif 'content' in response:
                content = response.get('content', '')
                json_data = self.parse_content(content)

            response['parse_info'] = json_data           
            response['vclip_num'] = len(json_data)
        self.logger('info', '✅ response 解析完毕 ...')
        return response
    
    def __call__(self, video_file, prompt):
        parse_info = {"success": True, "error": "", "response_time": 0.0}
        try:
            ## step1. 压缩
            comp_file = self.compress_video(video_file, 480)

            ## step2. 元数据
            video_info = self.get_video_info(comp_file)
            
            ## step3. base64
            self.encode_video_to_base64(video_info)
            
            ## step4. 装填
            payload = self.generate_payload(video_info, prompt)

            ## step5. 请求
            response = self.request_api(payload)

            ## step6. 解析
            parse_info = self.parse_response(response)
            
        except Exception as e:
            parse_info['success'] = False
            parse_info['error'] = f"分析过程中出错: {str(e)}"
            self.logger('error', '❌ 解析视频过程报错 : %s' % str(e))
        
        return parse_info