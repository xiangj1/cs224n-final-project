for file in ./dataset/*.flac;
do
    ffmpeg -i "$file" "${file%.*}.wav" -hide_banner -loglevel panic
done