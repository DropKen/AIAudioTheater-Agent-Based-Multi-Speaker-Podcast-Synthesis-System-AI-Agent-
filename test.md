 python assemble_podcast.py --seg_dir outputs\segments --out outputs\episode.mp3 `
>>   --ffmpeg "C:\Users\28799\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin\ffmpeg.exe"

 python indextts_headless.py --model_dir checkpoints --text "我今天心情非常好！" --emo_mode ref --emo_ref examples/happy_ref.wav --emo_weight 0.7 --out outputs/happy.wav   
 
set DEEPSEEK_API_KEY="sk-99511331e5ff449c9848f1138ce55660"

 python indextts_headless.py --model_dir checkpoints --prompt_audio examples python assemble_podcast.py --seg_dir outputs\segments --out outputs\episode.mp3 `
>>   --ffmpeg "C:\Users\28799\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin\ffmpeg.exe"
\voice_01.wav  --text "我今天心情非常好！" --emo_mode ref --emo_ref examples/voice_10.wav --emo_weight 0.7 --out outputs/happy.wav